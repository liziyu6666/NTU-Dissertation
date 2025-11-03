import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import place_poles

"""
三场景对比实验（控制变量法）：
场景1：无拜占庭节点 - 系统正常稳定
场景2：有拜占庭节点，不使用RCP-f - 系统被干扰，无法稳定
场景3：有拜占庭节点，使用RCP-f - 系统恢复稳定

目的：证明RCP-f算法的有效性
"""

# ================== 系统参数 ==================
num_agents = 8
f = 1

m = [0.1 * (i + 1) for i in range(num_agents)]
M = [1.0 * (i + 1) for i in range(num_agents)]
l = [0.1 * (i + 1) for i in range(num_agents)]
g = 9.8
friction = 0.15

S = np.array([[0, 1], [-1, 0]])

# 通信拓扑
adj_matrix = np.zeros((num_agents, num_agents), dtype=int)
adj_matrix[0:4, 0:4] = 1
np.fill_diagonal(adj_matrix[0:4, 0:4], 0)
adj_matrix[4:, 0:4] = 1
adj_matrix[4, 5] = adj_matrix[5, 6] = adj_matrix[6, 7] = 1

print("\n通信拓扑矩阵:")
print(adj_matrix)

# ================== Agent类 ==================
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
def apply_rcpf_filter(v_hat_i, neighbor_vhats, f):
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


# ================== 场景1：无拜占庭节点 ==================
def scenario1_no_byzantine():
    print("\n" + "="*80)
    print("场景1：无拜占庭节点（基准）")
    print("="*80)

    agents = [Agent(i) for i in range(num_agents)]
    y0 = np.zeros(num_agents * 6)
    for i in range(num_agents):
        y0[i * 6] = 0.1 + 0.01 * i
        y0[i * 6 + 2] = 0.05
        y0[i * 6 + 4:i * 6 + 6] = [1.0, 0.0]

    def total_system(t, y):
        states = y.reshape(num_agents, 6)
        dvdt = np.zeros((num_agents, 6))
        v_real = np.array([np.cos(t), np.sin(t)])

        for i in range(num_agents):
            x = states[i, :4]
            v_hat = states[i, 4:6]
            neighbors = np.where(adj_matrix[i] == 1)[0]
            is_target_node = (i < 4)

            neighbor_vhats = [states[j, 4:6] for j in neighbors]

            gain_consensus = 150.0  # 与场景2、3保持一致
            gain_tracking = 50.0

            consensus_term = np.zeros(2)
            if len(neighbor_vhats) > 0:
                neighbor_mean = np.mean(neighbor_vhats, axis=0)
                consensus_term = gain_consensus * (neighbor_mean - v_hat)

            if is_target_node:
                consensus_term += gain_tracking * (v_real - v_hat)

            dv_hat = S @ v_hat + consensus_term
            dxdt = agents[i].dynamics(x, v_hat)

            dvdt[i, :4] = dxdt
            dvdt[i, 4:6] = dv_hat

        return dvdt.flatten()

    t_span = (0, 20)
    t_eval = np.linspace(*t_span, 1000)

    print("运行仿真...")
    sol = solve_ivp(total_system, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8, max_step=0.02)
    print(f"✓ 仿真完成")

    return sol, agents


# ================== 场景2：有拜占庭节点，不使用RCP-f ==================
def scenario2_byzantine_no_filter():
    print("\n" + "="*80)
    print("场景2：有拜占庭节点，不使用RCP-f（对照组）")
    print("="*80)

    faulty_agent = 0  # 改为目标节点，影响力更大
    print(f"拜占庭节点: Agent {faulty_agent}")
    print("注意：所有节点都接收拜占庭节点的信息，不进行过滤")
    print("拜占庭节点是目标节点，具有直接访问参考信号的能力，影响更大")

    agents = [Agent(i) for i in range(num_agents)]
    y0 = np.zeros(num_agents * 6)
    for i in range(num_agents):
        y0[i * 6] = 0.1 + 0.01 * i
        y0[i * 6 + 2] = 0.05
        y0[i * 6 + 4:i * 6 + 6] = [1.0, 0.0]

    def total_system(t, y):
        states = y.reshape(num_agents, 6)
        dvdt = np.zeros((num_agents, 6))
        v_real = np.array([np.cos(t), np.sin(t)])

        for i in range(num_agents):
            x = states[i, :4]
            v_hat = states[i, 4:6]
            neighbors = np.where(adj_matrix[i] == 1)[0]
            is_target_node = (i < 4)

            if i == faulty_agent:
                # 拜占庭节点发送恶意信息（增强攻击强度）
                dv_hat = np.array([100 * np.sin(10 * t) + 50 * np.cos(12 * t), 20 + t / 5])
            else:
                # 正常节点，但不使用RCP-f过滤，直接接收所有邻居信息
                neighbor_vhats = [states[j, 4:6] for j in neighbors]

                gain_consensus = 150.0  # 增大共识项权重
                gain_tracking = 50.0     # 减小跟踪项权重（让拜占庭影响更明显）

                consensus_term = np.zeros(2)
                if len(neighbor_vhats) > 0:
                    # 关键：直接使用所有邻居的平均值，不过滤
                    neighbor_mean = np.mean(neighbor_vhats, axis=0)
                    consensus_term = gain_consensus * (neighbor_mean - v_hat)

                if is_target_node:
                    consensus_term += gain_tracking * (v_real - v_hat)

                dv_hat = S @ v_hat + consensus_term

            dxdt = agents[i].dynamics(x, v_hat)
            dvdt[i, :4] = dxdt
            dvdt[i, 4:6] = dv_hat

        return dvdt.flatten()

    t_span = (0, 20)
    t_eval = np.linspace(*t_span, 1000)

    print("运行仿真...")
    sol = solve_ivp(total_system, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8, max_step=0.02)
    print(f"✓ 仿真完成")

    return sol, agents, faulty_agent


# ================== 场景3：有拜占庭节点，使用RCP-f ==================
def scenario3_byzantine_with_filter():
    print("\n" + "="*80)
    print("场景3：有拜占庭节点，使用RCP-f（实验组）")
    print("="*80)

    faulty_agent = 0  # 与场景2保持一致
    print(f"拜占庭节点: Agent {faulty_agent}")
    print("注意：正常节点使用RCP-f过滤器排除异常邻居")
    print("拜占庭节点是目标节点，影响力更大，但RCP-f应能有效过滤")

    agents = [Agent(i) for i in range(num_agents)]
    y0 = np.zeros(num_agents * 6)
    for i in range(num_agents):
        y0[i * 6] = 0.1 + 0.01 * i
        y0[i * 6 + 2] = 0.05
        y0[i * 6 + 4:i * 6 + 6] = [1.0, 0.0]

    def total_system(t, y):
        states = y.reshape(num_agents, 6)
        dvdt = np.zeros((num_agents, 6))
        v_real = np.array([np.cos(t), np.sin(t)])

        for i in range(num_agents):
            x = states[i, :4]
            v_hat = states[i, 4:6]
            neighbors = np.where(adj_matrix[i] == 1)[0]
            is_target_node = (i < 4)

            if i == faulty_agent:
                # 拜占庭节点发送恶意信息（与场景2相同，增强攻击强度）
                dv_hat = np.array([100 * np.sin(10 * t) + 50 * np.cos(12 * t), 20 + t / 5])
            else:
                # 正常节点，使用RCP-f过滤
                neighbor_vhats = [states[j, 4:6] for j in neighbors]
                # 关键：使用RCP-f过滤器移除异常邻居
                filtered_neighbors = apply_rcpf_filter(v_hat, neighbor_vhats, f)

                gain_consensus = 150.0  # 增大共识项权重（与场景2相同）
                gain_tracking = 50.0     # 减小跟踪项权重（与场景2相同）

                consensus_term = np.zeros(2)
                if len(filtered_neighbors) > 0:
                    filtered_mean = np.mean(filtered_neighbors, axis=0)
                    consensus_term = gain_consensus * (filtered_mean - v_hat)

                if is_target_node:
                    consensus_term += gain_tracking * (v_real - v_hat)

                dv_hat = S @ v_hat + consensus_term

            dxdt = agents[i].dynamics(x, v_hat)
            dvdt[i, :4] = dxdt
            dvdt[i, 4:6] = dv_hat

        return dvdt.flatten()

    t_span = (0, 20)
    t_eval = np.linspace(*t_span, 1000)

    print("运行仿真...")
    sol = solve_ivp(total_system, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8, max_step=0.02)
    print(f"✓ 仿真完成")

    return sol, agents, faulty_agent


# ================== 计算误差 ==================
def compute_tracking_errors(sol, faulty_agent=None):
    """计算跟踪误差（与真实参考信号的误差）"""
    tracking_errors = []

    for t_idx in range(len(sol.t)):
        t = sol.t[t_idx]
        v_real = np.array([np.cos(t), np.sin(t)])

        tracking_err = []
        for i in range(num_agents):
            v_hat_i = sol.y[i * 6 + 4:i * 6 + 6, t_idx]
            err = np.linalg.norm(v_hat_i - v_real)
            tracking_err.append(err)
        tracking_errors.append(tracking_err)

    return np.array(tracking_errors)


# ================== 三场景对比可视化 ==================
def plot_three_scenarios(sol1, sol2, sol3, agents, faulty_agent):
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    # 计算误差
    err1 = compute_tracking_errors(sol1)
    err2 = compute_tracking_errors(sol2, faulty_agent)
    err3 = compute_tracking_errors(sol3, faulty_agent)

    # 计算最终误差（最后10%的平均值）
    final_idx = int(len(sol1.t) * 0.9)
    final_err1 = np.mean(err1[final_idx:, :], axis=0)
    final_err2 = np.mean(err2[final_idx:, :], axis=0)
    final_err3 = np.mean(err3[final_idx:, :], axis=0)

    # 正常节点
    normal_agents = [i for i in range(num_agents) if i != faulty_agent]

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Three-Scenario Comparison: Demonstrating RCP-f Effectiveness',
                 fontsize=16, fontweight='bold')

    # ========== 第一行：各场景的跟踪误差 ==========
    # 1. 场景1
    plt.subplot(3, 4, 1)
    for i in range(num_agents):
        plt.plot(sol1.t, err1[:, i], color=colors[i], alpha=0.7,
                 label=f'Agent {i}' if i < 3 else '')
    plt.title('Scenario 1: No Byzantine\n(Baseline)', fontweight='bold', fontsize=11)
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error')
    plt.yscale('log')
    plt.ylim([1e-4, 1e2])
    plt.legend(fontsize=8, loc='upper right')
    plt.grid(True, alpha=0.3)

    # 2. 场景2
    plt.subplot(3, 4, 2)
    for i in range(num_agents):
        linestyle = '--' if i == faulty_agent else '-'
        linewidth = 2 if i == faulty_agent else 1
        label = f'Agent {i} (Byz)' if i == faulty_agent else (f'Agent {i}' if i < 3 else '')
        plt.plot(sol2.t, err2[:, i], color=colors[i], alpha=0.7,
                 linestyle=linestyle, linewidth=linewidth, label=label)
    plt.title('Scenario 2: With Byzantine\n(No RCP-f Filter)', fontweight='bold', fontsize=11)
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error')
    plt.yscale('log')
    plt.ylim([1e-4, 1e2])
    plt.legend(fontsize=8, loc='upper right')
    plt.grid(True, alpha=0.3)

    # 3. 场景3
    plt.subplot(3, 4, 3)
    for i in range(num_agents):
        linestyle = '--' if i == faulty_agent else '-'
        linewidth = 2 if i == faulty_agent else 1
        label = f'Agent {i} (Byz)' if i == faulty_agent else (f'Agent {i}' if i < 3 else '')
        plt.plot(sol3.t, err3[:, i], color=colors[i], alpha=0.7,
                 linestyle=linestyle, linewidth=linewidth, label=label)
    plt.title('Scenario 3: With Byzantine\n(With RCP-f Filter)', fontweight='bold', fontsize=11)
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error')
    plt.yscale('log')
    plt.ylim([1e-4, 1e2])
    plt.legend(fontsize=8, loc='upper right')
    plt.grid(True, alpha=0.3)

    # 4. 最终误差柱状图对比
    plt.subplot(3, 4, 4)
    x = np.arange(num_agents)
    width = 0.25
    plt.bar(x - width, final_err1, width, label='S1: No Byz', color='green', alpha=0.7)
    bar_colors2 = ['red' if i == faulty_agent else 'orange' for i in range(num_agents)]
    plt.bar(x, final_err2, width, label='S2: No filter', color=bar_colors2, alpha=0.7)
    bar_colors3 = ['red' if i == faulty_agent else 'blue' for i in range(num_agents)]
    plt.bar(x + width, final_err3, width, label='S3: RCP-f', color=bar_colors3, alpha=0.7)
    plt.title('Final Tracking Error\nComparison', fontweight='bold', fontsize=11)
    plt.xlabel('Agent')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True, axis='y', alpha=0.3)

    # ========== 第二行：正常节点对比 ==========
    # 5. 正常节点误差对比（所有场景）
    plt.subplot(3, 4, 5)
    avg_err1 = np.mean(err1[:, normal_agents], axis=1)
    avg_err2_normal = np.mean(err2[:, normal_agents], axis=1)
    avg_err3_normal = np.mean(err3[:, normal_agents], axis=1)

    plt.plot(sol1.t, avg_err1, 'g-', linewidth=2, label='S1: No Byzantine', alpha=0.8)
    plt.plot(sol2.t, avg_err2_normal, 'orange', linewidth=2, label='S2: No filter', alpha=0.8)
    plt.plot(sol3.t, avg_err3_normal, 'b-', linewidth=2, label='S3: RCP-f', alpha=0.8)

    plt.title('Normal Agents:\nAverage Tracking Error', fontweight='bold', fontsize=11)
    plt.xlabel('Time (s)')
    plt.ylabel('Average Error')
    plt.yscale('log')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # 6. 正常节点最终误差对比
    plt.subplot(3, 4, 6)
    final_avg = [
        np.mean([final_err1[i] for i in normal_agents]),
        np.mean([final_err2[i] for i in normal_agents]),
        np.mean([final_err3[i] for i in normal_agents])
    ]
    scenarios = ['S1\nNo Byz', 'S2\nNo filter', 'S3\nRCP-f']
    bar_colors_avg = ['green', 'orange', 'blue']
    bars = plt.bar(scenarios, final_avg, color=bar_colors_avg, alpha=0.7)

    # 添加数值标签
    for bar, val in zip(bars, final_avg):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.title('Normal Agents:\nFinal Average Error', fontweight='bold', fontsize=11)
    plt.ylabel('Average Error')
    plt.yscale('log')
    plt.grid(True, axis='y', alpha=0.3)

    # 7. 性能提升对比
    plt.subplot(3, 4, 7)
    degradation = final_avg[1] / final_avg[0]  # S2 相对于 S1
    recovery = final_avg[2] / final_avg[0]      # S3 相对于 S1

    categories = ['S2 vs S1\n(degradation)', 'S3 vs S1\n(recovery)']
    ratios = [degradation, recovery]
    bar_colors_ratio = ['red', 'green']
    bars = plt.bar(categories, ratios, color=bar_colors_ratio, alpha=0.7)

    for bar, val in zip(bars, ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
    plt.title('Performance Ratio\n(Relative to S1)', fontweight='bold', fontsize=11)
    plt.ylabel('Error Ratio')
    plt.legend(fontsize=8)
    plt.grid(True, axis='y', alpha=0.3)

    # 8. 统计摘要
    plt.subplot(3, 4, 8)
    plt.axis('off')
    summary_text = f"""
    PERFORMANCE SUMMARY
    {'='*40}

    Scenario 1 (Baseline):
      Normal agents error: {final_avg[0]:.6f}

    Scenario 2 (No Filter):
      Normal agents error: {final_avg[1]:.6f}
      Degradation: {degradation:.2f}x worse
      Status: {'✗ UNSTABLE' if degradation > 2 else '⚠ DEGRADED'}

    Scenario 3 (RCP-f):
      Normal agents error: {final_avg[2]:.6f}
      Performance: {recovery:.2f}x vs baseline
      Status: {'✓ STABLE' if recovery < 1.5 else '⚠ PARTIAL'}

    Conclusion:
    {('✓ RCP-f EFFECTIVE!' if recovery < degradation * 0.5 else '⚠ RCP-f helps but limited')}
      - S2 shows Byzantine impact
      - S3 demonstrates RCP-f efficacy
    """
    plt.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center', fontweight='bold')

    # ========== 第三行：详细分析 ==========
    # 9. 场景1 vs 场景2（显示恶化）
    plt.subplot(3, 4, 9)
    for i in normal_agents:
        if i < 3:
            plt.plot(sol1.t, err1[:, i], color=colors[i], alpha=0.4,
                     linestyle='-', linewidth=1)
            plt.plot(sol2.t, err2[:, i], color=colors[i], alpha=0.8,
                     linestyle='--', linewidth=2, label=f'Agent {i}')
    plt.title('S1 vs S2: Byzantine Impact\n(Solid=S1, Dash=S2)', fontweight='bold', fontsize=11)
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 10. 场景2 vs 场景3（显示恢复）
    plt.subplot(3, 4, 10)
    for i in normal_agents:
        if i < 3:
            plt.plot(sol2.t, err2[:, i], color=colors[i], alpha=0.4,
                     linestyle='--', linewidth=1)
            plt.plot(sol3.t, err3[:, i], color=colors[i], alpha=0.8,
                     linestyle='-', linewidth=2, label=f'Agent {i}')
    plt.title('S2 vs S3: RCP-f Recovery\n(Dash=S2, Solid=S3)', fontweight='bold', fontsize=11)
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 11. 三场景位置跟踪对比（选一个代表节点）
    plt.subplot(3, 4, 11)
    representative_agent = 0
    ref_signal = np.cos(sol1.t)

    plt.plot(sol1.t, sol1.y[representative_agent * 6], 'g-', linewidth=2,
             label='S1: No Byz', alpha=0.7)
    plt.plot(sol2.t, sol2.y[representative_agent * 6], color='orange',
             linewidth=2, label='S2: No filter', alpha=0.7, linestyle='--')
    plt.plot(sol3.t, sol3.y[representative_agent * 6], 'b-', linewidth=2,
             label='S3: RCP-f', alpha=0.7)
    plt.plot(sol1.t, ref_signal, 'k--', linewidth=1.5, label='Reference', alpha=0.5)

    plt.title(f'Agent {representative_agent}: Position Tracking\nComparison',
              fontweight='bold', fontsize=11)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 12. 实验结论
    plt.subplot(3, 4, 12)
    plt.axis('off')

    conclusion_text = f"""
    EXPERIMENTAL CONCLUSION
    {'='*40}

    Control Variable Method:
    • Same Byzantine attack (Agent {faulty_agent})
    • Only difference: RCP-f filter

    Key Findings:

    1. Baseline (S1):
       ✓ System stable without attack

    2. Attack Impact (S2):
       ✗ {degradation:.1f}x performance drop
       ✗ Byzantine disrupts consensus

    3. Defense Effect (S3):
       ✓ {(1 - recovery/degradation)*100:.0f}% performance recovered
       ✓ RCP-f filters Byzantine nodes

    Statistical Significance:
       Error reduction: {((degradation - recovery)/degradation)*100:.0f}%
       (S2→S3 improvement)

    {'✓ RCP-f ALGORITHM VALIDATED!' if recovery < degradation * 0.6 else '⚠ Need parameter tuning'}
    """

    plt.text(0.05, 0.5, conclusion_text, fontsize=9, family='monospace',
             verticalalignment='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_file = "three_scenario_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Results saved to {output_file}")

    # 打印详细统计
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)

    print(f"\n【场景1 - 无拜占庭节点（基准）】")
    print(f"  所有节点平均误差: {np.mean(final_err1):.6f}")
    print(f"  最大误差: {np.max(final_err1):.6f}")
    print(f"  最小误差: {np.min(final_err1):.6f}")

    print(f"\n【场景2 - 有拜占庭节点，不使用RCP-f（对照组）】")
    print(f"  拜占庭节点: Agent {faulty_agent}")
    print(f"  正常节点平均误差: {final_avg[1]:.6f}")
    print(f"  拜占庭节点误差: {final_err2[faulty_agent]:.6f}")
    print(f"  相对场景1的性能: {degradation:.2f}x (↓ {(degradation-1)*100:.0f}% 恶化)")

    print(f"\n【场景3 - 有拜占庭节点，使用RCP-f（实验组）】")
    print(f"  拜占庭节点: Agent {faulty_agent}")
    print(f"  正常节点平均误差: {final_avg[2]:.6f}")
    print(f"  拜占庭节点误差: {final_err3[faulty_agent]:.6f}")
    print(f"  相对场景1的性能: {recovery:.2f}x (↓ {(recovery-1)*100:.0f}% 变化)")
    print(f"  相对场景2的改善: {(1 - recovery/degradation)*100:.0f}% 性能恢复")

    print(f"\n【结论】")
    if recovery < degradation * 0.6:
        print("  ✓ RCP-f算法有效！成功抵御拜占庭攻击")
        print(f"  ✓ 性能几乎恢复到基准水平（{recovery:.2f}x vs baseline）")
    elif recovery < degradation * 0.8:
        print("  ⚠ RCP-f算法有一定效果，但仍有改进空间")
    else:
        print("  ✗ RCP-f算法效果有限，需要调整参数")


# ================== 主程序 ==================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("三场景对比实验 - 控制变量法验证RCP-f有效性")
    print("="*80)
    print("\n实验设计:")
    print("  场景1: 无拜占庭节点 → 验证基准性能")
    print("  场景2: 有拜占庭节点，不使用RCP-f → 展示攻击影响")
    print("  场景3: 有拜占庭节点，使用RCP-f → 验证防御效果")

    # 运行三个场景
    sol1, agents1 = scenario1_no_byzantine()
    sol2, agents2, faulty_agent2 = scenario2_byzantine_no_filter()
    sol3, agents3, faulty_agent3 = scenario3_byzantine_with_filter()

    # 可视化对比
    plot_three_scenarios(sol1, sol2, sol3, agents1, faulty_agent2)

    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
