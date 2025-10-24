"""
修改版的仿真代码 - 用于收集完整的特征数据
用于训练LSTM模型检测拜占庭节点
"""

import numpy as np
import pickle
from scipy.integrate import solve_ivp
from scipy.signal import place_poles
import sys

# ================== 可配置参数 ==================
FAULTY_AGENT = 4  # 将被外部脚本修改
ATTACK_TYPE = 'mixed'  # 将被外部脚本修改
SCENARIO_ID = 0  # 将被外部脚本修改

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

# ================== 通信拓扑 ==================
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

        # 系统矩阵
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
                Q = np.eye(4) * 10.0
                R = np.array([[1.0]])
                from scipy.linalg import solve_continuous_are
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

    def get_control_input(self, x, v_hat):
        return self.K11 @ x + self.K12.flatten() @ v_hat


# ================== 拜占庭攻击 ==================
def byzantine_attack(t, attack_type):
    """不同类型的拜占庭攻击"""
    if attack_type == 'mixed':
        return np.array([50 * np.sin(10 * t) + 15 * np.cos(12 * t), t / 15])
    elif attack_type == 'sine':
        return np.array([30 * np.sin(8 * t), 30 * np.cos(8 * t)])
    elif attack_type == 'ramp':
        return np.array([t * 2, -t * 1.5])
    elif attack_type == 'constant':
        return np.array([100 + 20*np.sin(t), -50 + 10*np.cos(t)])
    elif attack_type == 'random':
        np.random.seed(int(t * 1000) % 10000)
        return np.random.randn(2) * 20
    else:
        return np.array([0, 0])


# ================== RCP-f 滤波器 ==================
def apply_rcpf_filter(v_hat_i, neighbor_vhats, f):
    """RCP-f滤波器"""
    if len(neighbor_vhats) == 0:
        return np.array([]).reshape(0, 2)

    neighbor_vhats = np.array(neighbor_vhats)
    n_neighbors = len(neighbor_vhats)

    if n_neighbors <= 2 * f:
        return neighbor_vhats

    distances = np.linalg.norm(neighbor_vhats - v_hat_i, axis=1)
    sorted_indices = np.argsort(distances)
    keep_indices = sorted_indices[:n_neighbors - f]

    return neighbor_vhats[keep_indices]


# ================== 数据收集 ==================
# 用于收集特征的全局变量
feature_data = {
    'time': [],
    'agents': [[] for _ in range(num_agents)]
}

def total_system(t, y):
    """系统动力学"""
    states = y.reshape(num_agents, 6)
    dvdt = np.zeros((num_agents, 6))
    v_real = np.array([np.cos(t), np.sin(t)])

    for i in range(num_agents):
        x = states[i, :4]
        v_hat = states[i, 4:6]
        neighbors = np.where(adj_matrix[i] == 1)[0]
        is_target_node = (i < 4)

        if i == FAULTY_AGENT:
            # 拜占庭智能体
            dv_hat = byzantine_attack(t, ATTACK_TYPE)
        else:
            # 正常智能体
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

        # 系统状态更新
        dxdt = agents[i].dynamics(x, v_hat)
        dvdt[i, :4] = dxdt
        dvdt[i, 4:6] = dv_hat

        # 收集特征数据（关键！）
        u = agents[i].get_control_input(x, v_hat)
        estimation_error = np.linalg.norm(v_hat - v_real)
        position_error = np.abs(x[0] - np.cos(t))

        feature_data['agents'][i].append({
            'time': t,
            'estimation_error': estimation_error,
            'position_error': position_error,
            'angle': x[2],
            'angular_velocity': x[3],
            'control_input': u,
            'v_hat_0': v_hat[0],
            'v_hat_1': v_hat[1],
        })

    # 记录时间
    if len(feature_data['time']) == 0 or abs(feature_data['time'][-1] - t) > 0.01:
        feature_data['time'].append(t)

    return dvdt.flatten()


# ================== 运行仿真 ==================
def run_simulation(faulty_agent, attack_type, scenario_id, silent=False):
    """运行单个仿真场景"""
    global FAULTY_AGENT, ATTACK_TYPE, SCENARIO_ID, agents, feature_data

    FAULTY_AGENT = faulty_agent
    ATTACK_TYPE = attack_type
    SCENARIO_ID = scenario_id

    # 重置feature_data
    feature_data = {
        'time': [],
        'agents': [[] for _ in range(num_agents)]
    }

    # 初始化智能体
    agents = [Agent(i) for i in range(num_agents)]

    # 初始条件
    y0 = np.zeros(num_agents * 6)
    for i in range(num_agents):
        y0[i * 6] = 0.1 + 0.01 * i
        y0[i * 6 + 2] = 0.05
        y0[i * 6 + 4:i * 6 + 6] = [1.0, 0.0]

    # 仿真
    t_span = (0, 15)
    t_eval = np.linspace(*t_span, 750)

    if not silent:
        print(f"场景 {scenario_id}: 拜占庭节点={faulty_agent}, 攻击={attack_type}")

    sol = solve_ivp(total_system, t_span, y0, t_eval=t_eval, method='RK45',
                   rtol=1e-6, atol=1e-8, max_step=0.02)

    if sol.status != 0:
        if not silent:
            print(f"  ✗ 仿真失败: {sol.message}")
        return None

    # 整理特征数据
    scenario_data = {
        'scenario_id': scenario_id,
        'faulty_agent': faulty_agent,
        'attack_type': attack_type,
        'time': np.array(feature_data['time']),
        'agents': []
    }

    for i in range(num_agents):
        agent_features = feature_data['agents'][i]

        # 转换为numpy数组
        agent_data = {
            'agent_id': i,
            'is_byzantine': (i == faulty_agent),
            'time': np.array([f['time'] for f in agent_features]),
            'estimation_error': np.array([f['estimation_error'] for f in agent_features]),
            'position_error': np.array([f['position_error'] for f in agent_features]),
            'angle': np.array([f['angle'] for f in agent_features]),
            'angular_velocity': np.array([f['angular_velocity'] for f in agent_features]),
            'control_input': np.array([f['control_input'] for f in agent_features]),
            'v_hat_0': np.array([f['v_hat_0'] for f in agent_features]),
            'v_hat_1': np.array([f['v_hat_1'] for f in agent_features]),
        }

        scenario_data['agents'].append(agent_data)

    if not silent:
        print(f"  ✓ 完成，采集 {len(scenario_data['time'])} 个时间点")

    return scenario_data


# ================== 主函数 ==================
if __name__ == '__main__':
    # 测试运行
    print("测试特征收集代码...")

    test_data = run_simulation(faulty_agent=4, attack_type='mixed', scenario_id=0)

    if test_data is not None:
        print("\n✓ 特征收集测试成功！")
        print(f"  - 时间点数: {len(test_data['time'])}")
        print(f"  - 智能体数: {len(test_data['agents'])}")
        print(f"  - 拜占庭节点: Agent {test_data['faulty_agent']}")

        # 显示一个智能体的特征形状
        agent0 = test_data['agents'][0]
        print(f"\nAgent 0 特征形状:")
        print(f"  - estimation_error: {agent0['estimation_error'].shape}")
        print(f"  - position_error: {agent0['position_error'].shape}")
        print(f"  - control_input: {agent0['control_input'].shape}")

        # 保存测试数据
        with open('test_scenario_data.pkl', 'wb') as f:
            pickle.dump(test_data, f)
        print(f"\n✓ 测试数据已保存至 test_scenario_data.pkl")
    else:
        print("✗ 特征收集测试失败")
