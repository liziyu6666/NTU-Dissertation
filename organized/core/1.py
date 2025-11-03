import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_sylvester
from scipy.signal import place_poles

# ================== 系统参数 ==================
num_agents = 8
faulty_agent = 4
f = 1  # 最大容忍拜占庭节点数

# 物理参数（每个智能体不同）- 论文Table 1
m = [0.1 * (i + 1) for i in range(num_agents)]  # mi = 0.1 * i kg
M = [1.0 * (i + 1) for i in range(num_agents)]  # Mi = i kg
l = [0.1 * (i + 1) for i in range(num_agents)]  # li = 0.1 * i m
g = 9.8  # 重力加速度
friction = 0.15  # 摩擦系数

# 参考信号动力学
S = np.array([[0, 1], [-1, 0]])  # 论文中的 v(t) 参考信号  用于 dvi(t) = S*vi(t)

# ================== 通信拓扑 ==================
# 前四个智能体完全连接，后四个与前四个有连接
adj_matrix = np.zeros((num_agents, num_agents), dtype=int)
adj_matrix[0:4, 0:4] = 1
np.fill_diagonal(adj_matrix[0:4, 0:4], 0)
adj_matrix[4:, 0:4] = 1
adj_matrix[4, 5] = adj_matrix[5, 6] = adj_matrix[6, 7] = 1

# ================== 代理类定义（异构系统） ==================
# 动力学方程为 dxi = Aixi + Biui + Eivi
class Agent:
    def __init__(self, index):
        self.index = index

        # 根据论文公式 (32)-(33) 和状态变换构建系统矩阵
        # 状态: x = [x1, x2, x3, x4] = [hi + li*θi, d(hi+li*θi)/dt, θi, dθi/dt]
        mi = m[index]
        Mi = M[index]
        li = l[index]
        fi = friction

        # 计算系统矩阵元素
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

        # Ei 矩阵：外部信号如何影响系统
        pi1 = 2.0 / Mi
        pi2 = 1.0 / (li * Mi)
        self.E = np.array([[0, 0], [pi1, 0], [0, 0], [pi2, 0]])

        # 输出矩阵
        self.C = np.array([[1, 0, -li, 0]])  # ei = hi - v01 = (hi + li*θi) - li*θi - v01
        self.F = np.array([[-1, 0]])

        # 求解调节方程 (Lemma 3 in paper):
        # Xi*S = Ai*Xi + Bi*Ui + Ei  (状态方程)
        # 0 = Ci*Xi + Fi              (输出方程)

        # 注意：这里的Ei对应论文中的-Ei*Gamma（因为参考信号动力学是dv=S*v）
        # 输出方程：ei = Ci*xi + Fi*v，我们要让ei->0，即Ci*Xi + Fi = 0

        try:
            # 使用Sylvester方程求解方法
            # 从 Xi*S = A*Xi + B*Ui + E 和 0 = C*Xi + F
            # 得到 Ui = -(C*Xi + F) 的关系不对，应该用增广方程

            # 正确方法：将调节方程重新组织
            # Xi @ S - A @ Xi = B @ Ui + E
            # C @ Xi = -F

            # 方法1：先用输出约束C @ Xi = -F求Xi的通解
            # 然后代入状态方程求Ui

            # C @ Xi + F = 0
            # [1, 0, -li, 0] @ Xi + [-1, 0] = 0
            # 即：Xi[0, :] - li*Xi[2, :] = [1, 0]

            # 设Xi的参数化形式：Xi[2,:] = [a, b], Xi[3,:] = [c, d]
            # 则 Xi[0,:] = [1, 0] + li*[a, b] = [1+li*a, li*b]
            # Xi[1,:] = [e, f] 是自由变量

            # 将Xi代入 Xi @ S - A @ Xi - E = B @ Ui
            # 这会给出关于(a,b,c,d,e,f)和Ui的线性方程

            # 更简洁的方法：直接求解增广系统的最小二乘解
            q = S.shape[0]  # = 2
            n = self.A.shape[0]  # = 4
            m_ctrl = self.B.shape[1]  # = 1 (控制输入维度，避免与全局变量m冲突)
            p = self.C.shape[0]  # = 1

            # 使用Kronecker积形式（这是标准方法）
            # vec(Xi @ S) = (S^T ⊗ I_n) vec(Xi)
            # vec(A @ Xi) = (I_q ⊗ A) vec(Xi)
            # vec(B @ Ui) = (I_q ⊗ B) vec(Ui)

            # 状态方程: (S^T ⊗ I_n) vec(Xi) - (I_q ⊗ A) vec(Xi) = (I_q ⊗ B) vec(Ui) + vec(E)
            # 即: [(S^T ⊗ I_n) - (I_q ⊗ A)] vec(Xi) - (I_q ⊗ B) vec(Ui) = vec(E)

            # 输出方程: (I_q ⊗ C) vec(Xi) = -vec(F)

            I_n = np.eye(n)
            I_q = np.eye(q)

            # 注意Kronecker积的顺序！
            A11 = np.kron(S.T, I_n) - np.kron(I_q, self.A)  # (nq) x (nq)
            A12 = -np.kron(I_q, self.B)                      # (nq) x (m_ctrl*q)
            A21 = np.kron(I_q, self.C)                       # (pq) x (nq)
            A22 = np.zeros((p*q, m_ctrl*q))                  # (pq) x (m_ctrl*q)

            # 组装矩阵
            A_top = np.hstack([A11, A12])
            A_bot = np.hstack([A21, A22])
            A_mat = np.vstack([A_top, A_bot])

            # 右侧向量
            b_top = self.E.flatten('F')  # Fortran order (column-major)
            b_bot = -self.F.flatten('F')
            b_vec = np.concatenate([b_top, b_bot])

            # 求解（使用伪逆，因为系统可能欠定）
            solution, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)

            # 提取解
            self.Xi = solution[:n*q].reshape((n, q), order='F')  # Fortran order
            self.Ui = solution[n*q:].reshape((m_ctrl, q), order='F')

            # 验证
            residual1 = self.Xi @ S - self.A @ self.Xi - self.B @ self.Ui - self.E
            residual2 = self.C @ self.Xi + self.F

            if np.linalg.norm(residual1) > 1e-4:
                print(f"Agent {index}: 状态方程残差 = {np.linalg.norm(residual1):.6e}")
            if np.linalg.norm(residual2) > 1e-4:
                print(f"Agent {index}: 输出方程残差 = {np.linalg.norm(residual2):.6e}")

        except Exception as e:
            print(f"Agent {index}: 调节方程求解失败 - {e}")
            import traceback
            traceback.print_exc()
            self.Xi = np.zeros((4, 2))
            self.Ui = np.zeros((1, 2))

        # 设计反馈增益 Ki1 使得 A_bar = Ai + Bi*Ki1 是 Hurwitz
        # 检查可控性：构造可控性矩阵 [B, AB, A^2B, A^3B]
        ctrl_matrix = np.hstack([
            self.B,
            self.A @ self.B,
            self.A @ self.A @ self.B,
            self.A @ self.A @ self.A @ self.B
        ])
        rank = np.linalg.matrix_rank(ctrl_matrix)

        if rank == 4:  # 完全可控
            # 使用LQR设计替代极点配置（place_poles数值不稳定）
            Q = np.diag([100.0, 10.0, 100.0, 10.0])  # 状态权重：位置和角度权重更大
            R = np.array([[0.1]])  # 控制输入权重
            from scipy.linalg import solve_continuous_are
            try:
                P = solve_continuous_are(self.A, self.B, Q, R)
                self.K11 = -(np.linalg.inv(R) @ self.B.T @ P).flatten()

                # 验证闭环稳定性
                A_closed = self.A + self.B @ self.K11.reshape(1, -1)
                eigenvalues = np.linalg.eigvals(A_closed)
                if np.any(eigenvalues.real >= 0):
                    print(f"Agent {index}: LQR产生不稳定闭环，使用手动增益")
                    self.K11 = np.array([-10 - 2*index, -5 - index, -50 - 5*index, -10 - 2*index])
            except Exception as e:
                print(f"Agent {index}: LQR失败 - {e}，使用手动增益")
                self.K11 = np.array([-10 - 2*index, -5 - index, -50 - 5*index, -10 - 2*index])
        else:
            print(f"Agent {index}: 系统不完全可控 (rank={rank})，使用手动增益")
            self.K11 = np.array([-100 - 20*index, -50 - 10*index,
                                  -200 - 40*index, -50 - 10*index])

        # Ki2 = Ui - Ki1*Xi
        self.K12 = self.Ui - self.K11.reshape(1, -1) @ self.Xi

    def dynamics(self, x, v_hat):
        u = self.K11 @ x + self.K12.flatten() @ v_hat
        return (self.A @ x + self.B.flatten() * u + self.E @ v_hat).flatten()


# ================== 弹性观测器（RCP-f 算法）==================
def apply_rcpf_filter(v_hat_i, neighbor_vhats, f, is_target_node=False):
    """
    实现论文中的 RCP-f 算法（Resilient Consensus Protocol with parameter f）
    基于欧氏距离移除最远的 f 个邻居值（作为整体向量）

    参数:
        v_hat_i: 当前智能体的估计值 (2,)
        neighbor_vhats: 邻居的估计值列表
        f: 最大容忍拜占庭节点数
        is_target_node: 是否为目标节点（能直接观测到参考信号）

    返回:
        filtered_neighbors: 过滤后的邻居值数组
    """
    if len(neighbor_vhats) == 0:
        return np.array([]).reshape(0, len(v_hat_i))

    neighbor_vhats = np.array(neighbor_vhats)
    n_neighbors = len(neighbor_vhats)

    # 如果邻居数量不足2f+1，无法有效过滤
    if n_neighbors <= 2 * f:
        return neighbor_vhats

    # 计算每个邻居与当前估计值的欧氏距离
    distances = np.linalg.norm(neighbor_vhats - v_hat_i, axis=1)

    # 按距离排序，移除最远的f个邻居
    sorted_indices = np.argsort(distances)
    # 保留距离最近的 (n - f) 个邻居
    keep_indices = sorted_indices[:n_neighbors - f]

    filtered_neighbors = neighbor_vhats[keep_indices]

    return filtered_neighbors


# ================== 计算总系统动态 ==================
vhat_diff_log = []
vhat_diff_time_log = []  # 记录对应时间点

def total_system(t, y):
    states = y.reshape(num_agents, 6)
    dvdt = np.zeros((num_agents, 6))
    v_real = np.array([np.cos(t), np.sin(t)])

    current_diff = []

    for i in range(num_agents):
        x = states[i, :4]
        v_hat = states[i, 4:6]
        neighbors = np.where(adj_matrix[i] == 1)[0]

        # 判断是否为目标节点（前4个节点）
        is_target_node = (i < 4)

        if i == faulty_agent:
            # 拜占庭智能体发送恶意信息（有界正弦信号攻击）
            dv_hat = np.array([50 * np.sin(10 * t) + 15 * np.cos(12 * t),
                               5 * np.sin(3 * t)])  # 修改为有界振荡，避免线性增长导致失稳
            current_diff.append(1000.0)  # 拜占庭节点标记为大值
        else:
            # 正常智能体使用弹性观测器
            neighbor_vhats = [states[j, 4:6] for j in neighbors]

            # 应用 RCP-f 过滤器（基于欧氏距离移除最远的f个邻居）
            filtered_neighbors = apply_rcpf_filter(v_hat, neighbor_vhats, f, is_target_node)

            # 计算观测器更新：dv_hat = S*v_hat + gain * sum(filtered_neighbors - v_hat)
            # 这里 gain 对应论文中的 d_ij，增大增益以提高收敛速度
            gain = 50.0  # 增大增益值

            # 计算共识项
            consensus_term = np.zeros(2)
            if len(filtered_neighbors) > 0:
                # 计算过滤后邻居的平均值
                filtered_mean = np.mean(filtered_neighbors, axis=0)
                consensus_term = gain * (filtered_mean - v_hat)

            # 如果是目标节点，添加来自参考信号的信息（更强的增益）
            if is_target_node:
                consensus_term += gain * 2.0 * (v_real - v_hat)

            # 论文公式 (15): dv_i = S*v_i + e^(St)*Phi_f(e^(-St)*v_j)
            # 简化版本：dv_hat = S*v_hat + consensus_term
            dv_hat = S @ v_hat + consensus_term

            # 记录与过滤后邻居均值的差异
            if len(filtered_neighbors) > 0:
                filtered_mean = np.mean(filtered_neighbors, axis=0)
                diff_norm = np.linalg.norm(v_hat - filtered_mean)
                current_diff.append(diff_norm)
            else:
                current_diff.append(0.0)

        # 计算系统状态更新
        dxdt = agents[i].dynamics(x, v_hat)
        dvdt[i, :4] = dxdt
        dvdt[i, 4:6] = dv_hat

    # 仅在指定时间点记录（t_eval中的点）
    if np.any(np.isclose(t, t_eval, atol=1e-5)):
        vhat_diff_log.append(current_diff)
        vhat_diff_time_log.append(t)

    return dvdt.flatten()


# ================== 初始化智能体 ==================
agents = [Agent(i) for i in range(num_agents)]
y0 = np.zeros(num_agents * 6)
for i in range(num_agents):
    y0[i * 6] = 0.1 + 0.01 * i
    y0[i * 6 + 2] = 0.05
    y0[i * 6 + 4:i * 6 + 6] = [1.0, 0.0]

# ================== 运行仿真 ==================
print("开始仿真...")
print(f"- 智能体数量: {num_agents}")
print(f"- 拜占庭节点: Agent {faulty_agent}")
print(f"- 容忍度 f={f}")

t_span = (0, 15)  # 15秒仿真时间，观察更长时间的收敛
t_eval = np.linspace(*t_span, 750)  # 采样点数
print(f"- 仿真时间: {t_span[1]}秒, 采样点: {len(t_eval)}")

sol = solve_ivp(total_system, t_span, y0, t_eval=t_eval, method='RK45',
                rtol=1e-6, atol=1e-8, max_step=0.02)
print(f"✓ 仿真完成，状态: {sol.status}, 信息: {sol.message}")

# ================== 性能指标评估（第4步） ==================
position_error = []
estimation_error = []
for i in range(num_agents):
    pos_i = sol.y[i * 6]
    ref = np.cos(sol.t)
    position_error.append(np.abs(pos_i - ref))
    
    est_i = sol.y[i * 6 + 4:i * 6 + 6]
    true_v = np.vstack((np.cos(sol.t), np.sin(sol.t)))
    estimation_error.append(np.linalg.norm(est_i - true_v, axis=0))

# ================== 可视化 ==================
plt.figure(figsize=(16, 12))
colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

# 添加总标题
plt.suptitle(f'Resilient Cooperative Output Regulation: {num_agents} Agents with Agent {faulty_agent} Byzantine (f={f})',
             fontsize=14, fontweight='bold')

# 1. 位置跟踪
plt.subplot(2, 3, 1)
for i in range(num_agents):
    label = f'Agent {i} (Byzantine)' if i == faulty_agent else (f'Agent {i}' if i < 3 else '')
    plt.plot(sol.t, sol.y[i * 6], linestyle='--' if i == faulty_agent else '-',
             color=colors[i], alpha=0.7, label=label)
plt.plot(sol.t, np.cos(sol.t), 'k-', linewidth=2, label='Reference')
plt.title('Position Tracking (x1 = h + l*θ)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)

# 2. 角度跟踪
plt.subplot(2, 3, 2)
for i in range(num_agents):
    plt.plot(sol.t, sol.y[i * 6 + 2], linestyle='--' if i == faulty_agent else '-', color=colors[i])
plt.title('Pendulum Angles')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid(True)

# 3. 估计误差
plt.subplot(2, 3, 3)
for i in range(num_agents):
    label = f'Agent {i} (Byz)' if i == faulty_agent else (f'Agent {i}' if i < 3 else '')
    plt.plot(sol.t, estimation_error[i], color=colors[i], alpha=0.7, label=label)
plt.title('Estimation Error Norm ||v_hat - v||')
plt.xlabel('Time (s)')
plt.ylabel('Error Norm')
plt.yscale('log')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)

# 4. 控制输入
plt.subplot(2, 3, 4)
for i in range(num_agents):
    u_history = [agents[i].K11 @ y[i * 6:i * 6 + 4] + agents[i].K12.flatten() @ y[i * 6 + 4:i * 6 + 6]
                 for y in sol.y.T]
    plt.plot(sol.t, u_history, color=colors[i],
             linestyle='--' if i == faulty_agent else '-',
             label=f'Agent {i}' if i < 3 or i == faulty_agent else '')
plt.title('Control Input Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Control Force (N)')
plt.legend()
plt.grid(True)

# 5. 位置误差指标
plt.subplot(2, 3, 5)
for i in range(num_agents):
    label = f'Agent {i} (Byz)' if i == faulty_agent else (f'Agent {i}' if i < 3 else '')
    plt.plot(sol.t, position_error[i], label=label, color=colors[i], alpha=0.7)
plt.title('Position Error |x1 - cos(t)|')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)

# 6. v_hat 差异监控（观测器收敛性分析）
plt.subplot(2, 3, 6)
if len(vhat_diff_log) > 0:
    vhat_diff_log_np = np.array(vhat_diff_log)
    for i in range(num_agents):
        if i != faulty_agent:  # 只显示正常节点
            label = f'Agent {i}' if i < 3 else ''
            plt.plot(vhat_diff_time_log, vhat_diff_log_np[:, i], color=colors[i],
                     alpha=0.7, label=label)
plt.title('Observer Consensus (Normal Agents)')
plt.xlabel('Time (s)')
plt.ylabel('|| v_hat - filtered_mean ||')
plt.yscale('log')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)

plt.tight_layout()
# 保存图片而不是显示（避免GUI问题）
output_file = "resilient_cor_results.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ 仿真完成！结果已保存到 {output_file}")

# ================== 保存 v_hat 差值日志为 CSV 文件 ==================
csv_filename = "vhat_difference_log.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['time'] + [f'agent{i}' for i in range(num_agents)]
    writer.writerow(header)
    
    for t, row in zip(vhat_diff_time_log, vhat_diff_log):
        writer.writerow([t] + list(row))

print(f"✅ v_hat 差值日志已保存至 {csv_filename}")
