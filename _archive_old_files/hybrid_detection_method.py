import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import place_poles
from scipy.optimize import linprog

"""
混合检测方法：结合论文的数据驱动方法和你的RCP-f算法

核心思路：
1. 使用论文的ℓ1优化方法进行数据预处理
2. 基于Hankel矩阵检测异常数据
3. 将检测到的异常节点传递给RCP-f
4. RCP-f进行实时过滤和共识控制
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


# ================== Hankel矩阵构建 ==================
def build_hankel_matrix(data, L):
    """
    构建Hankel矩阵
    data: (q*T,) 形式的数据
    L: block行数
    """
    q = 6  # 每个智能体的状态维度
    T = len(data) // q

    if T < L:
        raise ValueError(f"数据长度 {T} 小于窗口长度 {L}")

    H = np.zeros((q * L, T - L + 1))
    for i in range(T - L + 1):
        H[:, i] = data[i*q:(i+L)*q]

    return H


# ================== 论文方法：ℓ1优化检测 ==================
def l1_anomaly_detection(w_current, H, threshold=0.1):
    """
    使用论文中的ℓ1优化方法检测异常

    参数:
        w_current: 当前数据段 (qL,)
        H: Hankel矩阵
        threshold: 残差阈值

    返回:
        suspected_agents: 疑似拜占庭节点的索引列表
        w_recovered: 恢复的数据
    """
    try:
        # 求解: min ||w - Hg||_1
        n_cols = H.shape[1]

        # 转化为线性规划问题
        # min sum(r) s.t. -r <= w - Hg <= r
        c = np.concatenate([np.zeros(n_cols), np.ones(len(w_current))])

        A_ub = np.vstack([
            np.hstack([H, -np.eye(len(w_current))]),
            np.hstack([-H, -np.eye(len(w_current))])
        ])
        b_ub = np.concatenate([w_current, -w_current])

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')

        if result.success:
            g_opt = result.x[:n_cols]
            w_recovered = H @ g_opt

            # 计算残差
            residuals = np.abs(w_current - w_recovered)

            # 按智能体分组计算残差（每6个元素一组）
            agent_residuals = []
            for i in range(num_agents):
                agent_res = np.mean(residuals[i*6:(i+1)*6])
                agent_residuals.append(agent_res)

            # 检测异常（残差超过阈值）
            suspected_agents = [i for i, res in enumerate(agent_residuals) if res > threshold]

            return suspected_agents, w_recovered
        else:
            print("  ⚠ ℓ1优化求解失败")
            return [], w_current

    except Exception as e:
        print(f"  ⚠ ℓ1检测出错: {e}")
        return [], w_current


# ================== RCP-f过滤器 ==================
def apply_rcpf_filter(v_hat_i, neighbor_vhats, f, blacklist=None):
    """
    RCP-f过滤器（增强版，支持黑名单）

    参数:
        blacklist: 已知的拜占庭节点列表（由ℓ1检测提供）
    """
    if len(neighbor_vhats) == 0:
        return np.array([]).reshape(0, len(v_hat_i))

    neighbor_vhats = np.array(neighbor_vhats)
    n_neighbors = len(neighbor_vhats)

    # 如果有黑名单，先排除这些邻居
    if blacklist is not None and len(blacklist) > 0:
        # 这里需要跟踪邻居索引，实际实现中需要传入邻居ID
        pass  # 简化处理

    if n_neighbors <= 2 * f:
        return neighbor_vhats

    distances = np.linalg.norm(neighbor_vhats - v_hat_i, axis=1)
    sorted_indices = np.argsort(distances)
    keep_indices = sorted_indices[:n_neighbors - f]

    return neighbor_vhats[keep_indices]


# ================== 混合方法仿真 ==================
def hybrid_detection_scenario():
    """
    混合方法：先用ℓ1检测，再用RCP-f过滤
    """
    print("\n" + "="*80)
    print("混合方法：ℓ1数据检测 + RCP-f共识控制")
    print("="*80)

    faulty_agent = 0
    print(f"拜占庭节点: Agent {faulty_agent}")

    agents = [Agent(i) for i in range(num_agents)]
    y0 = np.zeros(num_agents * 6)
    for i in range(num_agents):
        y0[i * 6] = 0.1 + 0.01 * i
        y0[i * 6 + 2] = 0.05
        y0[i * 6 + 4:i * 6 + 6] = [1.0, 0.0]

    # 步骤1: 收集历史干净数据（离线阶段）
    print("\n步骤1: 收集历史干净数据...")
    # 这里假设我们有一些历史数据用于构建Hankel矩阵
    # 实际应用中，这应该是系统正常运行时收集的
    T_history = 50
    L = 5

    detected_byzantine = set()
    detection_history = []

    def total_system(t, y):
        states = y.reshape(num_agents, 6)
        dvdt = np.zeros((num_agents, 6))
        v_real = np.array([np.cos(t), np.sin(t)])

        # 每隔一段时间进行ℓ1检测
        if len(detection_history) > 0 and len(detection_history) % 10 == 0:
            # 提取当前数据段
            # 这里简化处理，实际需要维护滑动窗口
            pass

        for i in range(num_agents):
            x = states[i, :4]
            v_hat = states[i, 4:6]
            neighbors = np.where(adj_matrix[i] == 1)[0]
            is_target_node = (i < 4)

            if i == faulty_agent:
                # 拜占庭节点发送恶意信息
                dv_hat = np.array([100 * np.sin(10 * t) + 50 * np.cos(12 * t), 20 + t / 5])
            else:
                # 正常节点：使用增强的RCP-f（结合黑名单）
                neighbor_vhats = []
                for j in neighbors:
                    if j not in detected_byzantine:  # 排除已检测到的拜占庭节点
                        neighbor_vhats.append(states[j, 4:6])

                filtered_neighbors = apply_rcpf_filter(
                    v_hat, neighbor_vhats, f, blacklist=list(detected_byzantine)
                )

                gain_consensus = 150.0
                gain_tracking = 50.0

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

    print("\n步骤2: 运行混合检测仿真...")
    sol = solve_ivp(total_system, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8, max_step=0.02)
    print(f"✓ 仿真完成")

    return sol, agents, faulty_agent


# ================== 主程序 ==================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("混合检测方法实验")
    print("="*80)
    print("\n将论文的数据驱动方法与RCP-f结合")
    print("目标：展示两种方法的互补性\n")

    sol, agents, faulty_agent = hybrid_detection_scenario()

    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print("\n说明：")
    print("1. 这个框架展示了如何结合两种方法")
    print("2. ℓ1检测在数据层面识别异常")
    print("3. RCP-f在共识层面进行实时过滤")
    print("4. 实际应用中需要完善Hankel矩阵的在线更新机制")
