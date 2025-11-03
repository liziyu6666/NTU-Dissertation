import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import place_poles

"""
改进版对比实验：
- 计算真实的跟踪误差（与参考信号的误差）
- 验证无拜占庭节点时误差能趋近0
- 验证有拜占庭节点但使用RCP-f后正常节点也能趋近0
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
    print("场景1：无拜占庭节点")
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

            neighbor_vhats = [states[j, 4:6] for j in neighbors]

            gain = 100.0  # 增大增益以加快收敛
            consensus_term = np.zeros(2)
            if len(neighbor_vhats) > 0:
                neighbor_mean = np.mean(neighbor_vhats, axis=0)
                consensus_term = gain * (neighbor_mean - v_hat)

            if is_target_node:
                consensus_term += gain * 5.0 * (v_real - v_hat)  # 增强目标节点的跟踪能力

            dv_hat = S @ v_hat + consensus_term
            dxdt = agents[i].dynamics(x, v_hat)

            dvdt[i, :4] = dxdt
            dvdt[i, 4:6] = dv_hat

        return dvdt.flatten()

    t_span = (0, 30)  # 延长到30秒
    t_eval = np.linspace(*t_span, 1500)  # 相应增加采样点

    print("运行仿真（30秒，增强增益）...")
    sol = solve_ivp(total_system_normal, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8, max_step=0.02)
    print(f"✓ 仿真完成")

    return sol, agents


# ================== 场景2：有拜占庭节点 ==================
def scenario2_with_byzantine():
    print("\n" + "="*80)
    print("场景2：有拜占庭节点（使用RCP-f）")
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
                # 正常节点使用RCP-f
                neighbor_vhats = [states[j, 4:6] for j in neighbors]
                filtered_neighbors = apply_rcpf_filter(v_hat, neighbor_vhats, f)

                gain = 100.0  # 增大增益以加快收敛
                consensus_term = np.zeros(2)
                if len(filtered_neighbors) > 0:
                    filtered_mean = np.mean(filtered_neighbors, axis=0)
                    consensus_term = gain * (filtered_mean - v_hat)

                if is_target_node:
                    consensus_term += gain * 5.0 * (v_real - v_hat)  # 增强目标节点的跟踪能力

                dv_hat = S @ v_hat + consensus_term

            dxdt = agents[i].dynamics(x, v_hat)
            dvdt[i, :4] = dxdt
            dvdt[i, 4:6] = dv_hat

        return dvdt.flatten()

    t_span = (0, 30)  # 延长到30秒
    t_eval = np.linspace(*t_span, 1500)  # 相应增加采样点

    print("运行仿真（30秒，增强增益）...")
    sol = solve_ivp(total_system_with_byzantine, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8, max_step=0.02)
    print(f"✓ 仿真完成")

    return sol, agents, faulty_agent


# ================== 计算性能指标 ==================
def compute_errors(sol, faulty_agent=None):
    """
    计算两种误差：
    1. 跟踪误差：每个节点的估计值与真实参考信号的误差
    2. 共识误差：正常节点之间估计值的分散程度
    """
    tracking_errors = []
    consensus_errors = []

    for t_idx in range(len(sol.t)):
        t = sol.t[t_idx]
        v_real = np.array([np.cos(t), np.sin(t)])

        # 计算跟踪误差（与真实信号的误差）
        tracking_err = []
        for i in range(num_agents):
            v_hat_i = sol.y[i * 6 + 4:i * 6 + 6, t_idx]
            err = np.linalg.norm(v_hat_i - v_real)
            tracking_err.append(err)
        tracking_errors.append(tracking_err)

        # 计算共识误差（节点之间的分散程度）
        if faulty_agent is not None:
            # 只统计正常节点
            normal_vhats = [sol.y[i * 6 + 4:i * 6 + 6, t_idx]
                           for i in range(num_agents) if i != faulty_agent]
            vhat_mean = np.mean(normal_vhats, axis=0)

            consensus_err = []
            for i in range(num_agents):
                v_hat_i = sol.y[i * 6 + 4:i * 6 + 6, t_idx]
                if i != faulty_agent:
                    err = np.linalg.norm(v_hat_i - vhat_mean)
                    consensus_err.append(err)
                else:
                    consensus_err.append(np.nan)
            consensus_errors.append(consensus_err)
        else:
            # 所有节点都正常
            vhats = [sol.y[i * 6 + 4:i * 6 + 6, t_idx] for i in range(num_agents)]
            vhat_mean = np.mean(vhats, axis=0)

            consensus_err = []
            for i in range(num_agents):
                v_hat_i = sol.y[i * 6 + 4:i * 6 + 6, t_idx]
                err = np.linalg.norm(v_hat_i - vhat_mean)
                consensus_err.append(err)
            consensus_errors.append(consensus_err)

    return np.array(tracking_errors), np.array(consensus_errors)


# ================== 可视化 ==================
def plot_detailed_comparison(sol1, sol2, agents, faulty_agent):
    # 计算误差
    track_err1, cons_err1 = compute_errors(sol1)
    track_err2, cons_err2 = compute_errors(sol2, faulty_agent)

    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Detailed Comparison: No Byzantine vs. With Byzantine (RCP-f)',
                 fontsize=16, fontweight='bold')

    # ========== 场景1 ==========
    # 1. 跟踪误差
    plt.subplot(3, 4, 1)
    for i in range(num_agents):
        plt.plot(sol1.t, track_err1[:, i], color=colors[i], alpha=0.7,
                 label=f'Agent {i}' if i < 3 else '')
    plt.title('Scenario 1: Tracking Error\n||v_hat - v_true||', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.ylim([1e-6, 1e2])
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 2. 共识误差
    plt.subplot(3, 4, 2)
    for i in range(num_agents):
        plt.plot(sol1.t, cons_err1[:, i], color=colors[i], alpha=0.7,
                 label=f'Agent {i}' if i < 3 else '')
    plt.title('Scenario 1: Consensus Error\n||v_hat - v_hat_mean||', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.ylim([1e-6, 1e2])
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 3. 最终跟踪误差柱状图
    plt.subplot(3, 4, 3)
    final_idx = int(len(sol1.t) * 0.9)
    final_track_err1 = np.mean(track_err1[final_idx:, :], axis=0)
    plt.bar(range(num_agents), final_track_err1, color=colors, alpha=0.7)
    plt.title('Scenario 1: Final Tracking Error\n(Last 10% avg)', fontweight='bold')
    plt.xlabel('Agent')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.grid(True, axis='y', alpha=0.3)

    # 4. 统计信息
    plt.subplot(3, 4, 4)
    plt.axis('off')
    stats_text = f"""
    Scenario 1 (No Byzantine)
    {'='*35}

    Tracking Error (to true signal):
      Mean: {np.mean(final_track_err1):.8f}
      Max:  {np.max(final_track_err1):.8f}
      Min:  {np.min(final_track_err1):.8f}

    Consensus Error (between agents):
      Mean: {np.mean(cons_err1[final_idx:, :]):.8f}
      Max:  {np.max(cons_err1[final_idx:, :]):.8f}

    Status: {'✓ Converged' if np.mean(final_track_err1) < 0.01 else '✗ Not converged'}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center')

    # ========== 场景2 ==========
    # 5. 跟踪误差
    plt.subplot(3, 4, 5)
    for i in range(num_agents):
        linestyle = '--' if i == faulty_agent else '-'
        label = f'Agent {i} (Byz)' if i == faulty_agent else (f'Agent {i}' if i < 3 else '')
        plt.plot(sol2.t, track_err2[:, i], color=colors[i], alpha=0.7,
                 linestyle=linestyle, label=label)
    plt.title('Scenario 2: Tracking Error\n||v_hat - v_true||', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.ylim([1e-6, 1e2])
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 6. 共识误差（只显示正常节点）
    plt.subplot(3, 4, 6)
    for i in range(num_agents):
        if i != faulty_agent:
            plt.plot(sol2.t, cons_err2[:, i], color=colors[i], alpha=0.7,
                     label=f'Agent {i}' if i < 3 else '')
    plt.title('Scenario 2: Consensus Error (Normal)\n||v_hat - v_hat_mean||',
              fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.ylim([1e-6, 1e2])
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 7. 最终跟踪误差柱状图
    plt.subplot(3, 4, 7)
    final_track_err2 = np.mean(track_err2[final_idx:, :], axis=0)
    bar_colors = ['red' if i == faulty_agent else colors[i] for i in range(num_agents)]
    plt.bar(range(num_agents), final_track_err2, color=bar_colors, alpha=0.7)
    plt.title('Scenario 2: Final Tracking Error\n(Last 10% avg)', fontweight='bold')
    plt.xlabel('Agent')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.grid(True, axis='y', alpha=0.3)

    # 8. 统计信息
    plt.subplot(3, 4, 8)
    plt.axis('off')
    normal_agents = [i for i in range(num_agents) if i != faulty_agent]
    final_track_err2_normal = np.mean([final_track_err2[i] for i in normal_agents])
    final_cons_err2_normal = np.nanmean(cons_err2[final_idx:, normal_agents])

    stats_text2 = f"""
    Scenario 2 (With Byzantine)
    {'='*35}

    Normal Agents Tracking Error:
      Mean: {final_track_err2_normal:.8f}
      Max:  {np.max([final_track_err2[i] for i in normal_agents]):.8f}
      Min:  {np.min([final_track_err2[i] for i in normal_agents]):.8f}

    Byzantine Agent {faulty_agent}:
      Error: {final_track_err2[faulty_agent]:.8f}

    Normal Consensus Error:
      Mean: {final_cons_err2_normal:.8f}

    Status: {'✓ Converged' if final_track_err2_normal < 0.01 else '✗ Not converged'}
    """
    plt.text(0.1, 0.5, stats_text2, fontsize=10, family='monospace',
             verticalalignment='center')

    # ========== 对比分析 ==========
    # 9. 跟踪误差对比（正常节点）
    plt.subplot(3, 4, 9)
    for i in normal_agents:
        plt.plot(sol1.t, track_err1[:, i], color=colors[i], alpha=0.4,
                 linestyle='-', linewidth=1)
        plt.plot(sol2.t, track_err2[:, i], color=colors[i], alpha=0.7,
                 linestyle='--', linewidth=2, label=f'Agent {i}' if i < 3 else '')
    plt.title('Normal Agents: Tracking Error Comparison\n(Solid=S1, Dash=S2)',
              fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.ylim([1e-6, 1e2])
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 10. 最终误差对比柱状图
    plt.subplot(3, 4, 10)
    x = np.arange(num_agents)
    width = 0.35
    plt.bar(x - width/2, final_track_err1, width, label='Scenario 1',
            color='blue', alpha=0.6)
    bar_colors2 = ['red' if i == faulty_agent else 'orange' for i in range(num_agents)]
    plt.bar(x + width/2, final_track_err2, width, label='Scenario 2',
            color=bar_colors2, alpha=0.6)
    plt.title('Final Tracking Error Comparison', fontweight='bold')
    plt.xlabel('Agent')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)

    # 11. 收敛性分析
    plt.subplot(3, 4, 11)
    # 计算平均跟踪误差随时间变化
    avg_track_err1 = np.mean(track_err1, axis=1)
    avg_track_err2_normal = np.mean(track_err2[:, normal_agents], axis=1)

    plt.plot(sol1.t, avg_track_err1, 'b-', linewidth=2, label='Scenario 1 (all)', alpha=0.7)
    plt.plot(sol2.t, avg_track_err2_normal, 'orange', linewidth=2,
             label='Scenario 2 (normal)', linestyle='--', alpha=0.7)
    plt.title('Average Tracking Error Over Time', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Average Error')
    plt.yscale('log')
    plt.ylim([1e-4, 1e1])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 12. 总结
    plt.subplot(3, 4, 12)
    plt.axis('off')

    improvement_ratio = final_track_err2_normal / np.mean(final_track_err1)

    summary_text = f"""
    EXPERIMENT SUMMARY
    {'='*35}

    Question: Can system converge to
    zero error without Byzantine nodes?

    Answer:
      Scenario 1 avg error: {np.mean(final_track_err1):.6f}
      {'✓ YES - Near zero!' if np.mean(final_track_err1) < 0.001 else '✗ NO - Still has error'}

    Question: Can RCP-f handle
    Byzantine nodes?

    Answer:
      Scenario 2 avg error: {final_track_err2_normal:.6f}
      Performance ratio: {improvement_ratio:.2f}x
      {'✓ YES - Successfully filtered!' if improvement_ratio < 2.0 else '✗ NO - Significant degradation'}

    Conclusion:
      {'✓ Both scenarios converged!' if np.mean(final_track_err1) < 0.001 and final_track_err2_normal < 0.002 else '⚠ Need longer simulation time'}
    """

    plt.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', fontweight='bold')

    plt.tight_layout()
    output_file = "improved_comparison_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Results saved to {output_file}")

    # 详细打印
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)

    print(f"\n【场景1 - 无拜占庭节点】")
    print(f"  跟踪误差（与真实信号v(t)的误差）:")
    print(f"    平均: {np.mean(final_track_err1):.8f}")
    print(f"    最大: {np.max(final_track_err1):.8f}")
    print(f"    最小: {np.min(final_track_err1):.8f}")
    print(f"  共识误差（节点之间的分散程度）:")
    print(f"    平均: {np.mean(cons_err1[final_idx:, :]):.8f}")

    print(f"\n【场景2 - 有拜占庭节点（使用RCP-f）】")
    print(f"  拜占庭节点: Agent {faulty_agent}")
    print(f"  正常节点跟踪误差:")
    print(f"    平均: {final_track_err2_normal:.8f}")
    print(f"    最大: {np.max([final_track_err2[i] for i in normal_agents]):.8f}")
    print(f"  拜占庭节点跟踪误差: {final_track_err2[faulty_agent]:.8f} (预期很大)")
    print(f"  正常节点共识误差: {final_cons_err2_normal:.8f}")

    print(f"\n【性能对比】")
    print(f"  误差比率: {improvement_ratio:.2f}x")
    print(f"  结论: ", end="")
    if np.mean(final_track_err1) < 0.001 and final_track_err2_normal < 0.002:
        print("✓ 两个场景都成功收敛到接近0的误差！")
    elif np.mean(final_track_err1) < 0.01:
        print("⚠ 场景1误差较小但未达到理想值，可能需要更长仿真时间或调整参数")
    else:
        print("✗ 误差仍然较大，系统可能未完全收敛")


# ================== 主程序 ==================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("改进版对比实验：跟踪误差 vs 共识误差")
    print("="*80)

    # 场景1
    sol1, agents1 = scenario1_no_byzantine()

    # 场景2
    sol2, agents2, faulty_agent = scenario2_with_byzantine()

    # 可视化分析
    plot_detailed_comparison(sol1, sol2, agents1, faulty_agent)

    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
