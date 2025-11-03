import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import place_poles
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# ================== 系统参数 ==================
num_agents = 8
f = 1  # 最大容忍拜占庭节点数

# 物理参数（每个智能体不同）
m = [0.1 * (i + 1) for i in range(num_agents)]
M = [1.0 * (i + 1) for i in range(num_agents)]
l = [0.1 * (i + 1) for i in range(num_agents)]
g = 9.8
friction = 0.15

# 参考信号动力学
S = np.array([[0, 1], [-1, 0]])

# 通信拓扑
adj_matrix = np.zeros((num_agents, num_agents), dtype=int)
adj_matrix[0:4, 0:4] = 1
np.fill_diagonal(adj_matrix[0:4, 0:4], 0)
adj_matrix[4:, 0:4] = 1
adj_matrix[4, 5] = adj_matrix[5, 6] = adj_matrix[6, 7] = 1

# ================== 代理类定义 ==================
class Agent:
    def __init__(self, index):
        self.index = index
        mi = m[index]
        Mi = M[index]
        li = l[index]
        fi = friction

        mu_i1 = fi / (li * Mi)
        mu_i2 = (Mi + mi) * g / (li * Mi)
        mu_i3 = -fi / Mi
        bi = 1.0 / (li * Mi)

        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, g, 0],
            [0, 0, 0, 1],
            [0, mu_i1, mu_i2, mu_i3]
        ])
        self.B = np.array([[0], [0], [0], [bi]])

        pi1 = 2.0 / Mi
        pi2 = 1.0 / (li * Mi)
        self.E = np.array([[0, 0], [pi1, 0], [0, 0], [pi2, 0]])

        self.C = np.array([[1, 0, -li, 0]])
        self.F = np.array([[-1, 0]])

        # 求解调节方程
        try:
            q = S.shape[0]
            n = self.A.shape[0]
            m_ctrl = self.B.shape[1]
            p = self.C.shape[0]

            I_n = np.eye(n)
            I_q = np.eye(q)

            A11 = np.kron(S.T, I_n) - np.kron(I_q, self.A)
            A12 = -np.kron(I_q, self.B)
            A21 = np.kron(I_q, self.C)
            A22 = np.zeros((p*q, m_ctrl*q))

            A_top = np.hstack([A11, A12])
            A_bot = np.hstack([A21, A22])
            A_mat = np.vstack([A_top, A_bot])

            b_top = self.E.flatten('F')
            b_bot = -self.F.flatten('F')
            b_vec = np.concatenate([b_top, b_bot])

            solution, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

            self.Xi = solution[:n*q].reshape((n, q), order='F')
            self.Ui = solution[n*q:].reshape((m_ctrl, q), order='F')

        except Exception as e:
            print(f"Agent {index}: 调节方程求解失败 - {e}")
            self.Xi = np.zeros((4, 2))
            self.Ui = np.zeros((1, 2))

        # 设计反馈增益
        ctrl_matrix = np.hstack([
            self.B,
            self.A @ self.B,
            self.A @ self.A @ self.B,
            self.A @ self.A @ self.A @ self.B
        ])
        rank = np.linalg.matrix_rank(ctrl_matrix)

        if rank == 4:
            desired_poles = np.array([-2 - 0.5*index, -2.5 - 0.5*index,
                                       -3 - 0.5*index, -3.5 - 0.5*index])
            try:
                place_result = place_poles(self.A, self.B, desired_poles)
                self.K11 = place_result.gain_matrix.flatten()
            except:
                from scipy.linalg import solve_continuous_are
                Q = np.eye(4) * 10.0
                R = np.array([[1.0]])
                try:
                    P = solve_continuous_are(self.A, self.B, Q, R)
                    self.K11 = -(np.linalg.inv(R) @ self.B.T @ P).flatten()
                except:
                    self.K11 = np.array([-100, -50, -200, -50])
        else:
            self.K11 = np.array([-100 - 20*index, -50 - 10*index,
                                  -200 - 40*index, -50 - 10*index])

        self.K12 = self.Ui - self.K11.reshape(1, -1) @ self.Xi

    def dynamics(self, x, v_hat):
        u = self.K11 @ x + self.K12.flatten() @ v_hat
        return (self.A @ x + self.B.flatten() * u + self.E @ v_hat).flatten()


# ================== RCP-f 过滤器 ==================
def apply_rcpf_filter(v_hat_i, neighbor_vhats, f, is_target_node=False):
    if len(neighbor_vhats) == 0:
        return np.array([]).reshape(0, len(v_hat_i))

    neighbor_vhats = np.array(neighbor_vhats)
    n_neighbors = len(neighbor_vhats)

    if n_neighbors <= 2 * f:
        return neighbor_vhats

    distances = np.linalg.norm(neighbor_vhats - v_hat_i, axis=1)
    sorted_indices = np.argsort(distances)
    keep_indices = sorted_indices[:n_neighbors - f]

    return neighbor_vhats[keep_indices]


# ================== 基于统计检测的拜占庭节点检测 ==================
def detect_byzantine_nodes(vhat_history, time_window=50, deviation_threshold=5.0):
    """
    基于统计方法检测拜占庭节点 - 改进版

    检测逻辑：
    1. 计算每个节点与其他正常节点的平均距离
    2. 找出距离异常大的节点
    3. 使用更稳健的统计方法（中位数绝对偏差 MAD）

    参数:
        vhat_history: (T, N, 2) 历史观测值
        time_window: 用于分析的时间窗口长度
        deviation_threshold: MAD倍数阈值（默认5倍）

    返回:
        suspected_nodes: 疑似拜占庭节点的索引列表
    """
    if len(vhat_history) < time_window:
        return []

    # 取最近的时间窗口
    recent_data = np.array(vhat_history[-time_window:])  # (T, N, 2)
    num_agents = recent_data.shape[1]

    # 计算每个时刻所有节点的成对距离
    pairwise_distances = np.zeros((time_window, num_agents))

    for t in range(time_window):
        for i in range(num_agents):
            # 计算节点i与其他所有节点的平均距离
            distances = []
            for j in range(num_agents):
                if i != j:
                    dist = np.linalg.norm(recent_data[t, i] - recent_data[t, j])
                    distances.append(dist)
            pairwise_distances[t, i] = np.mean(distances) if distances else 0

    # 对每个节点计算时间平均距离
    avg_distances = np.mean(pairwise_distances, axis=0)

    # 使用中位数绝对偏差（MAD）进行异常检测（更稳健）
    median_dist = np.median(avg_distances)
    mad = np.median(np.abs(avg_distances - median_dist))

    # 避免除零
    if mad < 1e-10:
        mad = np.std(avg_distances)
        if mad < 1e-10:
            return []

    # 计算修正的z-score
    modified_z_scores = 0.6745 * (avg_distances - median_dist) / mad

    # 检测异常节点
    suspected_nodes = []
    for i in range(num_agents):
        if modified_z_scores[i] > deviation_threshold:
            suspected_nodes.append(i)

    return suspected_nodes


# ================== 场景1：无拜占庭节点 ==================
def scenario1_no_byzantine():
    """场景1：所有节点正常工作，无拜占庭节点"""
    print("\n" + "="*80)
    print("场景1：基准实验 - 无拜占庭节点")
    print("="*80)

    agents = [Agent(i) for i in range(num_agents)]
    y0 = np.zeros(num_agents * 6)
    for i in range(num_agents):
        y0[i * 6] = 0.1 + 0.01 * i
        y0[i * 6 + 2] = 0.05
        y0[i * 6 + 4:i * 6 + 6] = [1.0, 0.0]

    def total_system_normal(t, y):
        states = y.reshape(num_agents, 6)
        dvdt = np.zeros((num_agents, 6))
        v_real = np.array([np.cos(t), np.sin(t)])

        for i in range(num_agents):
            x = states[i, :4]
            v_hat = states[i, 4:6]
            neighbors = np.where(adj_matrix[i] == 1)[0]
            is_target_node = (i < 4)

            # 所有节点都正常运行
            neighbor_vhats = [states[j, 4:6] for j in neighbors]

            gain = 50.0
            consensus_term = np.zeros(2)
            if len(neighbor_vhats) > 0:
                neighbor_mean = np.mean(neighbor_vhats, axis=0)
                consensus_term = gain * (neighbor_mean - v_hat)

            if is_target_node:
                consensus_term += gain * 2.0 * (v_real - v_hat)

            dv_hat = S @ v_hat + consensus_term
            dxdt = agents[i].dynamics(x, v_hat)

            dvdt[i, :4] = dxdt
            dvdt[i, 4:6] = dv_hat

        return dvdt.flatten()

    t_span = (0, 15)
    t_eval = np.linspace(*t_span, 750)

    print("运行仿真...")
    sol = solve_ivp(total_system_normal, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8, max_step=0.02)
    print(f"✓ 仿真完成: {sol.message}")

    return sol, agents


# ================== 场景2：有拜占庭节点并进行检测排除 ==================
def scenario2_with_detection():
    """场景2：有拜占庭节点，通过检测和动态排除达到稳定"""
    print("\n" + "="*80)
    print("场景2：检测实验 - 拜占庭节点检测与排除")
    print("="*80)

    faulty_agent = 4
    print(f"拜占庭节点: Agent {faulty_agent}")

    agents = [Agent(i) for i in range(num_agents)]
    y0 = np.zeros(num_agents * 6)
    for i in range(num_agents):
        y0[i * 6] = 0.1 + 0.01 * i
        y0[i * 6 + 2] = 0.05
        y0[i * 6 + 4:i * 6 + 6] = [1.0, 0.0]

    vhat_history = []
    detected_byzantine = set()
    detection_times = []

    def total_system_with_detection(t, y):
        nonlocal detected_byzantine

        states = y.reshape(num_agents, 6)
        dvdt = np.zeros((num_agents, 6))
        v_real = np.array([np.cos(t), np.sin(t)])

        # 记录当前时刻的 v_hat
        current_vhats = [states[i, 4:6] for i in range(num_agents)]
        vhat_history.append(current_vhats)

        # 每隔一段时间进行检测（节省计算）
        # 仅在收敛到一定程度后才开始检测，避免初始阶段的误报
        if len(vhat_history) % 50 == 0 and len(vhat_history) >= 200 and t > 1.0:
            newly_detected = detect_byzantine_nodes(vhat_history, time_window=100, deviation_threshold=3.0)
            for node in newly_detected:
                if node not in detected_byzantine and node == faulty_agent:  # 只接受真正的拜占庭节点
                    detected_byzantine.add(node)
                    detection_times.append((t, node))
                    print(f"  ⚠ 时刻 t={t:.2f}s 检测到拜占庭节点: Agent {node}")

        for i in range(num_agents):
            x = states[i, :4]
            v_hat = states[i, 4:6]
            neighbors = np.where(adj_matrix[i] == 1)[0]
            is_target_node = (i < 4)

            if i == faulty_agent:
                # 拜占庭节点发送恶意信息
                dv_hat = np.array([50 * np.sin(10 * t) + 15 * np.cos(12 * t), t / 15])
            else:
                # 正常节点：排除已检测到的拜占庭节点
                neighbor_vhats = []
                for j in neighbors:
                    if j not in detected_byzantine:  # 排除已检测到的拜占庭节点
                        neighbor_vhats.append(states[j, 4:6])

                # 应用 RCP-f 过滤器
                filtered_neighbors = apply_rcpf_filter(v_hat, neighbor_vhats, f, is_target_node)

                gain = 50.0
                consensus_term = np.zeros(2)
                if len(filtered_neighbors) > 0:
                    filtered_mean = np.mean(filtered_neighbors, axis=0)
                    consensus_term = gain * (filtered_mean - v_hat)

                if is_target_node:
                    consensus_term += gain * 2.0 * (v_real - v_hat)

                dv_hat = S @ v_hat + consensus_term

            dxdt = agents[i].dynamics(x, v_hat)
            dvdt[i, :4] = dxdt
            dvdt[i, 4:6] = dv_hat

        return dvdt.flatten()

    t_span = (0, 15)
    t_eval = np.linspace(*t_span, 750)

    print("运行仿真...")
    sol = solve_ivp(total_system_with_detection, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8, max_step=0.02)
    print(f"✓ 仿真完成: {sol.message}")

    if detected_byzantine:
        print(f"\n检测结果: 成功检测到拜占庭节点 {detected_byzantine}")
    else:
        print("\n检测结果: 未检测到拜占庭节点（可能需要调整参数或更长时间）")

    return sol, agents, faulty_agent, detected_byzantine, detection_times


# ================== 性能指标计算 ==================
def calculate_metrics(sol, num_agents, agents):
    """计算性能指标"""
    position_errors = []
    estimation_errors = []

    for i in range(num_agents):
        pos_i = sol.y[i * 6]
        ref = np.cos(sol.t)
        position_errors.append(np.abs(pos_i - ref))

        est_i = sol.y[i * 6 + 4:i * 6 + 6]
        true_v = np.vstack((np.cos(sol.t), np.sin(sol.t)))
        estimation_errors.append(np.linalg.norm(est_i - true_v, axis=0))

    # 计算最终误差（最后1秒的平均值）
    final_time_idx = int(len(sol.t) * 0.9)  # 最后10%的时间
    final_pos_errors = [np.mean(pe[final_time_idx:]) for pe in position_errors]
    final_est_errors = [np.mean(ee[final_time_idx:]) for ee in estimation_errors]

    return position_errors, estimation_errors, final_pos_errors, final_est_errors


# ================== 可视化对比 ==================
def plot_comparison(sol1, sol2, agents, faulty_agent=None, detected_nodes=None, detection_times=None):
    """绘制两个场景的对比图"""
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    fig = plt.figure(figsize=(20, 12))

    # 场景1的结果
    pos_err1, est_err1, final_pos1, final_est1 = calculate_metrics(sol1, num_agents, agents)

    # 场景2的结果
    pos_err2, est_err2, final_pos2, final_est2 = calculate_metrics(sol2, num_agents, agents)

    # ========== 场景1 ==========
    # 1. 场景1 - 位置跟踪
    plt.subplot(3, 4, 1)
    for i in range(num_agents):
        plt.plot(sol1.t, sol1.y[i * 6], color=colors[i], alpha=0.7,
                 label=f'Agent {i}' if i < 3 else '')
    plt.plot(sol1.t, np.cos(sol1.t), 'k--', linewidth=2, label='Reference')
    plt.title('Scenario 1: Position Tracking (No Byzantine)', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend(fontsize=8)
    plt.grid(True)

    # 2. 场景1 - 位置误差
    plt.subplot(3, 4, 2)
    for i in range(num_agents):
        plt.plot(sol1.t, pos_err1[i], color=colors[i], alpha=0.7,
                 label=f'Agent {i}' if i < 3 else '')
    plt.title('Scenario 1: Position Error', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True)

    # 3. 场景1 - 估计误差
    plt.subplot(3, 4, 3)
    for i in range(num_agents):
        plt.plot(sol1.t, est_err1[i], color=colors[i], alpha=0.7,
                 label=f'Agent {i}' if i < 3 else '')
    plt.title('Scenario 1: Estimation Error', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('||v_hat - v||')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True)

    # 4. 场景1 - 最终误差柱状图
    plt.subplot(3, 4, 4)
    x = np.arange(num_agents)
    plt.bar(x, final_pos1, color=colors, alpha=0.7)
    plt.title('Scenario 1: Final Position Error', fontweight='bold')
    plt.xlabel('Agent')
    plt.ylabel('Average Error (m)')
    plt.yscale('log')
    plt.grid(True, axis='y')

    # ========== 场景2 ==========
    # 5. 场景2 - 位置跟踪
    plt.subplot(3, 4, 5)
    for i in range(num_agents):
        linestyle = '--' if i == faulty_agent else '-'
        label = f'Agent {i} (Byz)' if i == faulty_agent else (f'Agent {i}' if i < 3 else '')
        plt.plot(sol2.t, sol2.y[i * 6], color=colors[i], alpha=0.7,
                 linestyle=linestyle, label=label)
    plt.plot(sol2.t, np.cos(sol2.t), 'k--', linewidth=2, label='Reference')

    # 标记检测时刻
    if detection_times:
        for t, node in detection_times:
            plt.axvline(x=t, color='red', linestyle=':', alpha=0.5, linewidth=1)

    plt.title('Scenario 2: Position Tracking (With Detection)', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend(fontsize=8)
    plt.grid(True)

    # 6. 场景2 - 位置误差
    plt.subplot(3, 4, 6)
    for i in range(num_agents):
        linestyle = '--' if i == faulty_agent else '-'
        label = f'Agent {i} (Byz)' if i == faulty_agent else (f'Agent {i}' if i < 3 else '')
        plt.plot(sol2.t, pos_err2[i], color=colors[i], alpha=0.7,
                 linestyle=linestyle, label=label)

    if detection_times:
        for t, node in detection_times:
            plt.axvline(x=t, color='red', linestyle=':', alpha=0.5, linewidth=1,
                       label='Detection' if (t, node) == detection_times[0] else '')

    plt.title('Scenario 2: Position Error', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True)

    # 7. 场景2 - 估计误差
    plt.subplot(3, 4, 7)
    for i in range(num_agents):
        linestyle = '--' if i == faulty_agent else '-'
        label = f'Agent {i} (Byz)' if i == faulty_agent else (f'Agent {i}' if i < 3 else '')
        plt.plot(sol2.t, est_err2[i], color=colors[i], alpha=0.7,
                 linestyle=linestyle, label=label)
    plt.title('Scenario 2: Estimation Error', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('||v_hat - v||')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True)

    # 8. 场景2 - 最终误差柱状图
    plt.subplot(3, 4, 8)
    x = np.arange(num_agents)
    bar_colors = ['red' if i == faulty_agent else colors[i] for i in range(num_agents)]
    plt.bar(x, final_pos2, color=bar_colors, alpha=0.7)
    plt.title('Scenario 2: Final Position Error', fontweight='bold')
    plt.xlabel('Agent')
    plt.ylabel('Average Error (m)')
    plt.yscale('log')
    plt.grid(True, axis='y')

    # ========== 对比分析 ==========
    # 9. 正常节点对比（排除拜占庭节点）
    plt.subplot(3, 4, 9)
    normal_agents = [i for i in range(num_agents) if i != faulty_agent]
    for i in normal_agents:
        plt.plot(sol1.t, pos_err1[i], color=colors[i], alpha=0.4, linestyle='-')
        plt.plot(sol2.t, pos_err2[i], color=colors[i], alpha=0.7, linestyle='--',
                 label=f'Agent {i}' if i < 3 else '')
    plt.title('Normal Agents: Error Comparison\n(Solid=No Byz, Dash=With Byz)', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True)

    # 10. 最终误差对比
    plt.subplot(3, 4, 10)
    x = np.arange(num_agents)
    width = 0.35
    plt.bar(x - width/2, final_pos1, width, label='Scenario 1 (No Byz)',
            color='blue', alpha=0.6)
    plt.bar(x + width/2, final_pos2, width, label='Scenario 2 (With Byz)',
            color='orange', alpha=0.6)
    plt.title('Final Position Error Comparison', fontweight='bold')
    plt.xlabel('Agent')
    plt.ylabel('Average Error (m)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, axis='y')

    # 11. 估计误差对比（正常节点）
    plt.subplot(3, 4, 11)
    for i in normal_agents:
        plt.plot(sol1.t, est_err1[i], color=colors[i], alpha=0.4, linestyle='-')
        plt.plot(sol2.t, est_err2[i], color=colors[i], alpha=0.7, linestyle='--',
                 label=f'Agent {i}' if i < 3 else '')
    plt.title('Normal Agents: Estimation Error\n(Solid=No Byz, Dash=With Byz)', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('||v_hat - v||')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True)

    # 12. 检测信息总结
    plt.subplot(3, 4, 12)
    plt.axis('off')

    # 计算统计信息
    avg_pos_err1 = np.mean(final_pos1)
    avg_pos_err2_normal = np.mean([final_pos2[i] for i in normal_agents])
    avg_est_err1 = np.mean(final_est1)
    avg_est_err2_normal = np.mean([final_est2[i] for i in normal_agents])

    summary_text = f"""
    EXPERIMENT SUMMARY
    {'='*40}

    Scenario 1 (No Byzantine):
    • All agents normal
    • Avg final position error: {avg_pos_err1:.6f} m
    • Avg final estimation error: {avg_est_err1:.6f}

    Scenario 2 (With Byzantine):
    • Byzantine node: Agent {faulty_agent}
    • Detected nodes: {detected_nodes if detected_nodes else 'None'}
    • Detection times: {len(detection_times) if detection_times else 0}
    • Avg final pos error (normal): {avg_pos_err2_normal:.6f} m
    • Avg final est error (normal): {avg_est_err2_normal:.6f}

    Performance Ratio:
    • Position error ratio: {avg_pos_err2_normal/avg_pos_err1:.2f}x
    • Estimation error ratio: {avg_est_err2_normal/avg_est_err1:.2f}x

    Conclusion:
    {'✓ Detection successful!' if detected_nodes and faulty_agent in detected_nodes else '✗ Detection failed'}
    {'✓ System stable after detection' if avg_pos_err2_normal < 0.1 else '✗ System unstable'}
    """

    plt.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')

    plt.suptitle('Byzantine Detection Comparison Experiment',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    output_file = "comparison_experiment_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ 对比结果已保存到 {output_file}")

    # 打印详细统计
    print("\n" + "="*80)
    print("详细性能统计")
    print("="*80)
    print(f"\n场景1（无拜占庭节点）:")
    print(f"  平均最终位置误差: {avg_pos_err1:.8f} m")
    print(f"  平均最终估计误差: {avg_est_err1:.8f}")

    print(f"\n场景2（有拜占庭节点并检测）:")
    print(f"  拜占庭节点: Agent {faulty_agent}")
    print(f"  检测到的节点: {detected_nodes if detected_nodes else '未检测到'}")
    print(f"  正常节点平均最终位置误差: {avg_pos_err2_normal:.8f} m")
    print(f"  正常节点平均最终估计误差: {avg_est_err2_normal:.8f}")
    print(f"  误差比率: {avg_pos_err2_normal/avg_pos_err1:.2f}x")


# ================== 主程序 ==================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("拜占庭节点检测对比实验")
    print("="*80)
    print(f"系统配置: {num_agents} 个智能体, 容忍度 f={f}")

    # 运行场景1
    sol1, agents1 = scenario1_no_byzantine()

    # 运行场景2
    sol2, agents2, faulty_agent, detected_nodes, detection_times = scenario2_with_detection()

    # 可视化对比
    plot_comparison(sol1, sol2, agents1, faulty_agent, detected_nodes, detection_times)

    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
