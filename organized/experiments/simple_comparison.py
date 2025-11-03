import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import place_poles

"""
简化版对比实验：
场景1：无拜占庭节点，所有节点达到共识（误差趋近0）
场景2：有拜占庭节点，通过RCP-f算法排除后，正常节点也能达到共识
"""

# ================== 系统参数 ==================
num_agents = 8
f = 1  # 最大容忍拜占庭节点数

# 物理参数（简化版，使用相同参数以便对比）
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

print("通信拓扑:")
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
    print("场景1：无拜占庭节点 - 所有节点正常协作")
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

            # 所有节点都正常
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
    print(f"✓ 仿真完成")

    # 后处理：计算共识误差
    consensus_errors = []
    for t_idx in range(len(sol.t)):
        vhats = [sol.y[i * 6 + 4:i * 6 + 6, t_idx] for i in range(num_agents)]
        vhat_mean = np.mean(vhats, axis=0)
        consensus_err = [np.linalg.norm(vhats[i] - vhat_mean) for i in range(num_agents)]
        consensus_errors.append(consensus_err)

    return sol, agents, np.array(consensus_errors)


# ================== 场景2：有拜占庭节点使用RCP-f ==================
def scenario2_with_byzantine():
    print("\n" + "="*80)
    print("场景2：有拜占庭节点 - 使用RCP-f算法排除")
    print("="*80)

    faulty_agent = 4
    print(f"拜占庭节点: Agent {faulty_agent}")

    agents = [Agent(i) for i in range(num_agents)]
    y0 = np.zeros(num_agents * 6)
    for i in range(num_agents):
        y0[i * 6] = 0.1 + 0.01 * i
        y0[i * 6 + 2] = 0.05
        y0[i * 6 + 4:i * 6 + 6] = [1.0, 0.0]

    def total_system_with_byzantine(t, y):
        states = y.reshape(num_agents, 6)
        dvdt = np.zeros((num_agents, 6))
        v_real = np.array([np.cos(t), np.sin(t)])

        for i in range(num_agents):
            x = states[i, :4]
            v_hat = states[i, 4:6]
            neighbors = np.where(adj_matrix[i] == 1)[0]
            is_target_node = (i < 4)

            if i == faulty_agent:
                # 拜占庭节点发送恶意信息
                dv_hat = np.array([50 * np.sin(10 * t) + 15 * np.cos(12 * t), t / 15])
            else:
                # 正常节点使用RCP-f过滤
                neighbor_vhats = [states[j, 4:6] for j in neighbors]
                filtered_neighbors = apply_rcpf_filter(v_hat, neighbor_vhats, f)

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
    sol = solve_ivp(total_system_with_byzantine, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8, max_step=0.02)
    print(f"✓ 仿真完成")

    # 后处理：计算共识误差（只统计正常节点）
    consensus_errors = []
    for t_idx in range(len(sol.t)):
        vhats = [sol.y[i * 6 + 4:i * 6 + 6, t_idx] for i in range(num_agents) if i != faulty_agent]
        vhat_mean = np.mean(vhats, axis=0)
        consensus_err = []
        for i in range(num_agents):
            if i != faulty_agent:
                consensus_err.append(np.linalg.norm(sol.y[i * 6 + 4:i * 6 + 6, t_idx] - vhat_mean))
            else:
                consensus_err.append(np.nan)  # 拜占庭节点不参与统计
        consensus_errors.append(consensus_err)

    return sol, agents, faulty_agent, np.array(consensus_errors)


# ================== 可视化 ==================
def plot_comparison(sol1, sol2, agents, consensus_err1, consensus_err2, faulty_agent):
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('拜占庭节点检测对比实验：RCP-f算法验证', fontsize=16, fontweight='bold')

    # 计算参考信号
    v_real = np.vstack([np.cos(sol1.t), np.sin(sol1.t)])

    # ========== 场景1 ==========
    # 1. 观测器估计（场景1）
    plt.subplot(2, 4, 1)
    for i in range(num_agents):
        est_i = sol1.y[i * 6 + 4:i * 6 + 6]
        est_error = np.linalg.norm(est_i - v_real, axis=0)
        plt.plot(sol1.t, est_error, color=colors[i], alpha=0.7,
                 label=f'Agent {i}' if i < 3 else '')
    plt.title('场景1: 观测器估计误差\n(无拜占庭节点)', fontweight='bold')
    plt.xlabel('时间 (s)')
    plt.ylabel('||v_hat - v||')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 2. 位置跟踪（场景1）
    plt.subplot(2, 4, 2)
    for i in range(num_agents):
        pos_i = sol1.y[i * 6]
        plt.plot(sol1.t, pos_i, color=colors[i], alpha=0.7,
                 label=f'Agent {i}' if i < 3 else '')
    plt.plot(sol1.t, np.cos(sol1.t), 'k--', linewidth=2, label='参考')
    plt.title('场景1: 位置跟踪', fontweight='bold')
    plt.xlabel('时间 (s)')
    plt.ylabel('位置 (m)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 3. 共识误差（场景1）
    plt.subplot(2, 4, 3)
    for i in range(num_agents):
        plt.plot(sol1.t, consensus_err1[:, i], color=colors[i], alpha=0.7,
                 label=f'Agent {i}' if i < 3 else '')
    plt.title('场景1: 共识误差', fontweight='bold')
    plt.xlabel('时间 (s)')
    plt.ylabel('||v_hat - v_hat_mean||')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 4. 最终误差统计（场景1）
    plt.subplot(2, 4, 4)
    final_idx = int(len(sol1.t) * 0.9)
    final_consensus_err1 = np.nanmean(consensus_err1[final_idx:, :], axis=0)
    plt.bar(range(num_agents), final_consensus_err1, color=colors, alpha=0.7)
    plt.title('场景1: 最终共识误差\n(最后10%时间平均)', fontweight='bold')
    plt.xlabel('智能体')
    plt.ylabel('平均共识误差')
    plt.yscale('log')
    plt.grid(True, axis='y', alpha=0.3)

    # ========== 场景2 ==========
    # 5. 观测器估计（场景2）
    plt.subplot(2, 4, 5)
    for i in range(num_agents):
        est_i = sol2.y[i * 6 + 4:i * 6 + 6]
        est_error = np.linalg.norm(est_i - v_real, axis=0)
        linestyle = '--' if i == faulty_agent else '-'
        label = f'Agent {i} (拜占庭)' if i == faulty_agent else (f'Agent {i}' if i < 3 else '')
        plt.plot(sol2.t, est_error, color=colors[i], alpha=0.7,
                 linestyle=linestyle, label=label)
    plt.title('场景2: 观测器估计误差\n(有拜占庭节点)', fontweight='bold')
    plt.xlabel('时间 (s)')
    plt.ylabel('||v_hat - v||')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 6. 位置跟踪（场景2）
    plt.subplot(2, 4, 6)
    for i in range(num_agents):
        pos_i = sol2.y[i * 6]
        linestyle = '--' if i == faulty_agent else '-'
        label = f'Agent {i} (拜占庭)' if i == faulty_agent else (f'Agent {i}' if i < 3 else '')
        plt.plot(sol2.t, pos_i, color=colors[i], alpha=0.7,
                 linestyle=linestyle, label=label)
    plt.plot(sol2.t, np.cos(sol2.t), 'k--', linewidth=2, label='参考')
    plt.title('场景2: 位置跟踪', fontweight='bold')
    plt.xlabel('时间 (s)')
    plt.ylabel('位置 (m)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 7. 共识误差（场景2，只显示正常节点）
    plt.subplot(2, 4, 7)
    for i in range(num_agents):
        if i != faulty_agent:
            plt.plot(sol2.t, consensus_err2[:, i], color=colors[i], alpha=0.7,
                     label=f'Agent {i}' if i < 3 else '')
    plt.title('场景2: 共识误差 (正常节点)', fontweight='bold')
    plt.xlabel('时间 (s)')
    plt.ylabel('||v_hat - v_hat_mean||')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 8. 最终误差对比
    plt.subplot(2, 4, 8)
    final_idx = int(len(sol2.t) * 0.9)
    final_consensus_err2 = np.nanmean(consensus_err2[final_idx:, :], axis=0)

    x = np.arange(num_agents)
    width = 0.35
    plt.bar(x - width/2, final_consensus_err1, width, label='场景1 (无拜占庭)',
            color='blue', alpha=0.6)
    bar_colors = ['red' if i == faulty_agent else 'orange' for i in range(num_agents)]
    plt.bar(x + width/2, final_consensus_err2, width, label='场景2 (有拜占庭)',
            color=bar_colors, alpha=0.6)
    plt.title('最终共识误差对比', fontweight='bold')
    plt.xlabel('智能体')
    plt.ylabel('平均共识误差')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = "simple_comparison_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ 结果已保存到 {output_file}")

    # 打印统计信息
    print("\n" + "="*80)
    print("性能统计")
    print("="*80)

    # 场景1统计
    final_consensus1 = np.mean(final_consensus_err1)
    print(f"\n场景1（无拜占庭节点）:")
    print(f"  平均最终共识误差: {final_consensus1:.8f}")
    print(f"  最大最终共识误差: {np.max(final_consensus_err1):.8f}")

    # 场景2统计（只统计正常节点）
    normal_agents = [i for i in range(num_agents) if i != faulty_agent]
    final_consensus2_normal = np.mean([final_consensus_err2[i] for i in normal_agents])
    print(f"\n场景2（有拜占庭节点，使用RCP-f）:")
    print(f"  拜占庭节点: Agent {faulty_agent}")
    print(f"  正常节点平均最终共识误差: {final_consensus2_normal:.8f}")
    print(f"  正常节点最大最终共识误差: {np.max([final_consensus_err2[i] for i in normal_agents]):.8f}")
    print(f"  拜占庭节点共识误差: {final_consensus_err2[faulty_agent]:.8f} (预期很大)")

    # 对比
    print(f"\n性能对比:")
    print(f"  误差比率（场景2/场景1）: {final_consensus2_normal/final_consensus1:.2f}x")

    if final_consensus2_normal < 0.1:
        print(f"\n✓ 结论: RCP-f算法成功排除拜占庭节点影响，正常节点达到共识！")
    else:
        print(f"\n✗ 结论: 正常节点误差仍较大，需要调整参数")


# ================== 主程序 ==================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("拜占庭节点检测对比实验 - 简化版")
    print("="*80)
    print(f"系统配置: {num_agents} 个智能体, 容忍度 f={f}")

    # 场景1
    sol1, agents1, consensus_err1 = scenario1_no_byzantine()

    # 场景2
    sol2, agents2, faulty_agent, consensus_err2 = scenario2_with_byzantine()

    # 可视化
    plot_comparison(sol1, sol2, agents1, consensus_err1, consensus_err2, faulty_agent)

    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
