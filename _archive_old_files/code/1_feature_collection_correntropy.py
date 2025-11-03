"""
改进版特征收集 - 添加Correntropy特征
基于MCA论文的思想，使用高斯核函数计算节点间相似度
"""

import numpy as np
import pickle
from scipy.integrate import solve_ivp
from scipy.signal import place_poles
import sys

# ================== 可配置参数 ==================
FAULTY_AGENT = 4
ATTACK_TYPE = 'mixed'
SCENARIO_ID = 0

# ================== 系统参数 ==================
num_agents = 8
f = 1  # 最大容忍拜占庭节点数

# 物理参数
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

        desired_poles = np.array([-5.0, -6.0, -7.0, -8.0])
        K = place_poles(self.A, self.B, desired_poles).gain_matrix

        self.K = K
        self.Li = (self.Xi - self.A @ self.Xi @ np.linalg.inv(S)).T

    def get_control_input(self, x, v_hat):
        u_ff = self.Ui @ v_hat
        u_fb = -self.K @ (x - self.Xi @ v_hat)
        return (u_ff + u_fb).item()

    def dynamics(self, x, v_hat):
        u = self.get_control_input(x, v_hat)
        return self.A @ x + self.B.flatten() * u


# ================== 辅助函数 ==================
def apply_rcpf_filter(v_hat_i, neighbor_vhats, f):
    """RCP-f过滤器（基于欧式距离）"""
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


def byzantine_attack(t, attack_type):
    """拜占庭攻击"""
    if attack_type == 'sine':
        return np.array([5.0 * np.sin(3 * t), 5.0 * np.cos(3 * t)])
    elif attack_type == 'constant':
        return np.array([10.0, 10.0])
    elif attack_type == 'random':
        return np.random.randn(2) * 5.0
    elif attack_type == 'ramp':
        return np.array([0.5 * t, 0.5 * t])
    elif attack_type == 'mixed':
        return np.array([5.0 * np.sin(3 * t) + 0.5 * t, 5.0 * np.cos(3 * t)])
    else:
        return np.zeros(2)


def compute_correntropy_features(v_hat_i, all_vhats, agent_i, sigma=1.0):
    """
    计算Correntropy特征（MCA论文核心思想）

    参数:
        v_hat_i: 当前agent的估计值 (2,)
        all_vhats: 所有agent的估计值列表 [(2,), (2,), ...]
        agent_i: 当前agent的索引
        sigma: 高斯核宽度

    返回:
        dict: {
            'avg_correntropy': 平均相似度,
            'min_correntropy': 最小相似度,
            'std_correntropy': 相似度标准差
        }

    原理:
        G_σ(x-y) = exp(-||x-y||²/(2σ²))
        - Normal agent: 与邻居相似 → correntropy高
        - Byzantine agent: 与邻居差异大 → correntropy低
    """
    if len(all_vhats) <= 1:
        return {
            'avg_correntropy': 0.0,
            'min_correntropy': 0.0,
            'std_correntropy': 0.0
        }

    corrs = []
    for j, v_hat_j in enumerate(all_vhats):
        if j != agent_i:  # 排除自己
            diff = np.linalg.norm(v_hat_i - v_hat_j)
            corr = np.exp(-diff**2 / (2 * sigma**2))
            corrs.append(corr)

    if len(corrs) == 0:
        return {
            'avg_correntropy': 0.0,
            'min_correntropy': 0.0,
            'std_correntropy': 0.0
        }

    return {
        'avg_correntropy': np.mean(corrs),
        'min_correntropy': np.min(corrs),
        'std_correntropy': np.std(corrs)
    }


# ================== 全局变量 ==================
agents = []
feature_data = {
    'time': [],
    'agents': [[] for _ in range(num_agents)]
}

def total_system(t, y):
    """系统动力学（改进版：在每个时间步计算correntropy特征）"""
    states = y.reshape(num_agents, 6)
    dvdt = np.zeros((num_agents, 6))
    v_real = np.array([np.cos(t), np.sin(t)])

    # 收集当前时刻所有agent的v_hat（用于计算correntropy）
    all_vhats = [states[j, 4:6] for j in range(num_agents)]

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

        # ========== 计算特征 ==========
        u = agents[i].get_control_input(x, v_hat)
        estimation_error = np.linalg.norm(v_hat - v_real)
        position_error = np.abs(x[0] - np.cos(t))

        # 🆕 计算Correntropy特征
        # 自适应sigma：基于当前所有v_hat的分散程度
        v_mean = np.mean(all_vhats, axis=0)
        distances = [np.linalg.norm(v - v_mean) for v in all_vhats]
        sigma_adaptive = max(np.median(distances), 0.1)  # 至少0.1避免除零

        corr_features = compute_correntropy_features(v_hat, all_vhats, i, sigma=sigma_adaptive)

        # 保存完整特征（7个原始 + 3个correntropy = 10个）
        feature_data['agents'][i].append({
            'time': t,
            # 原始7个特征
            'estimation_error': estimation_error,
            'position_error': position_error,
            'angle': x[2],
            'angular_velocity': x[3],
            'control_input': u,
            'v_hat_0': v_hat[0],
            'v_hat_1': v_hat[1],
            # 🆕 新增3个Correntropy特征
            'avg_correntropy': corr_features['avg_correntropy'],
            'min_correntropy': corr_features['min_correntropy'],
            'std_correntropy': corr_features['std_correntropy'],
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

        # 转换为numpy数组（10个特征）
        agent_data = {
            'agent_id': i,
            'is_byzantine': (i == faulty_agent),
            'time': np.array([f['time'] for f in agent_features]),
            # 原始7个特征
            'estimation_error': np.array([f['estimation_error'] for f in agent_features]),
            'position_error': np.array([f['position_error'] for f in agent_features]),
            'angle': np.array([f['angle'] for f in agent_features]),
            'angular_velocity': np.array([f['angular_velocity'] for f in agent_features]),
            'control_input': np.array([f['control_input'] for f in agent_features]),
            'v_hat_0': np.array([f['v_hat_0'] for f in agent_features]),
            'v_hat_1': np.array([f['v_hat_1'] for f in agent_features]),
            # 🆕 新增3个Correntropy特征
            'avg_correntropy': np.array([f['avg_correntropy'] for f in agent_features]),
            'min_correntropy': np.array([f['min_correntropy'] for f in agent_features]),
            'std_correntropy': np.array([f['std_correntropy'] for f in agent_features]),
        }

        scenario_data['agents'].append(agent_data)

    if not silent:
        print(f"  ✓ 完成，采集 {len(scenario_data['time'])} 个时间点，10维特征")

    return scenario_data


# ================== 主函数 ==================
if __name__ == '__main__':
    print("="*60)
    print("测试Correntropy特征收集")
    print("="*60)

    test_data = run_simulation(faulty_agent=4, attack_type='sine', scenario_id=0)

    if test_data is not None:
        print("\n✓ 特征收集测试成功！")
        print(f"  - 时间点数: {len(test_data['time'])}")
        print(f"  - 智能体数: {len(test_data['agents'])}")
        print(f"  - 拜占庭节点: Agent {test_data['faulty_agent']}")

        # 显示特征
        agent0 = test_data['agents'][0]
        print(f"\nAgent 0 特征列表:")
        for key in agent0.keys():
            if key not in ['agent_id', 'is_byzantine']:
                print(f"  - {key}: shape {agent0[key].shape}")

        # 分析Correntropy特征
        print(f"\nCorrentropy特征分析:")
        for i in range(num_agents):
            agent = test_data['agents'][i]
            avg_corr = np.mean(agent['avg_correntropy'])
            marker = "🔴 Byzantine" if agent['is_byzantine'] else "🟢 Normal"
            print(f"  Agent {i} {marker}: avg_correntropy = {avg_corr:.4f}")

        # 保存
        with open('test_correntropy_data.pkl', 'wb') as f:
            pickle.dump(test_data, f)
        print(f"\n✓ 测试数据已保存至 test_correntropy_data.pkl")
    else:
        print("✗ 特征收集测试失败")
