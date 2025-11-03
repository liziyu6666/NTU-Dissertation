#!/usr/bin/env python3
"""
测试核心逻辑：验证Agent类的初始化和RCP-f算法
"""
import numpy as np
from scipy.linalg import solve_sylvester
from scipy.signal import place_poles

# ================== 系统参数 ==================
num_agents = 8
f = 1  # 最大容忍拜占庭节点数

# 物理参数（每个智能体不同）- 论文Table 1
m = [0.1 * (i + 1) for i in range(num_agents)]  # mi = 0.1 * i kg
M = [1.0 * (i + 1) for i in range(num_agents)]  # Mi = i kg
l = [0.1 * (i + 1) for i in range(num_agents)]  # li = 0.1 * i m
g = 9.8  # 重力加速度
friction = 0.15  # 摩擦系数

# 参考信号动力学
S = np.array([[0, 1], [-1, 0]])

print("=" * 60)
print("测试1: 物理参数设置")
print("=" * 60)
for i in range(num_agents):
    print(f"Agent {i}: mi={m[i]:.2f} kg, Mi={M[i]:.2f} kg, li={l[i]:.2f} m")

# ================== 代理类定义 ==================
class Agent:
    def __init__(self, index):
        self.index = index

        # 根据论文公式 (32)-(33) 和状态变换构建系统矩阵
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
        self.C = np.array([[1, 0, -li, 0]])
        self.F = np.array([[-1, 0]])

        # 求解调节方程
        try:
            self.Xi = solve_sylvester(self.A, -S, -self.E)
            target = -self.A @ self.Xi - self.E - self.Xi @ S
            self.Ui = np.linalg.lstsq(self.B, target, rcond=None)[0]
        except Exception as e:
            print(f"Agent {index}: 调节方程求解失败 - {e}")
            self.Xi = np.zeros((4, 2))
            self.Ui = np.zeros((1, 2))

        # 设计反馈增益
        desired_poles = np.array([-2 - 0.5*index, -2.5 - 0.5*index,
                                   -3 - 0.5*index, -3.5 - 0.5*index])
        try:
            place_result = place_poles(self.A, self.B, desired_poles)
            self.K11 = place_result.gain_matrix.flatten()
        except Exception as e:
            print(f"Agent {index}: 极点配置失败 - {e}")
            self.K11 = np.array([-10, -20, -30, -10])

        # Ki2 = Ui - Ki1*Xi
        self.K12 = self.Ui - self.K11.reshape(1, -1) @ self.Xi

print("\n" + "=" * 60)
print("测试2: Agent类初始化")
print("=" * 60)

agents = []
for i in range(num_agents):
    try:
        agent = Agent(i)
        agents.append(agent)
        print(f"Agent {i}: 初始化成功")
        print(f"  A矩阵形状: {agent.A.shape}")
        print(f"  B矩阵形状: {agent.B.shape}")
        print(f"  E矩阵形状: {agent.E.shape}")
        print(f"  Xi形状: {agent.Xi.shape}")
        print(f"  K11形状: {agent.K11.shape}")
        print(f"  K12形状: {agent.K12.shape}")

        # 验证调节方程
        residual = agent.A @ agent.Xi + agent.Xi @ S + agent.E
        print(f"  调节方程残差范数: {np.linalg.norm(residual):.6f}")

        # 检查闭环稳定性
        A_cl = agent.A + agent.B @ agent.K11.reshape(1, -1)
        eigenvalues = np.linalg.eigvals(A_cl)
        print(f"  闭环极点实部: {[f'{e.real:.2f}' for e in eigenvalues]}")

    except Exception as e:
        print(f"Agent {i}: 初始化失败 - {e}")
        import traceback
        traceback.print_exc()

# ================== 测试 RCP-f 过滤器 ==================
def apply_rcpf_filter(v_hat_i, neighbor_vhats, f):
    """RCP-f 算法实现"""
    if len(neighbor_vhats) == 0:
        return [[] for _ in range(len(v_hat_i))]

    neighbor_vhats = np.array(neighbor_vhats)
    n_neighbors = len(neighbor_vhats)
    filtered_values = []

    # 对每个维度独立应用 MSR 算法
    for dim in range(len(v_hat_i)):
        dim_values = neighbor_vhats[:, dim]
        current_value = v_hat_i[dim]

        larger_indices = np.where(dim_values > current_value)[0]
        smaller_indices = np.where(dim_values < current_value)[0]

        # 移除极值
        if len(larger_indices) > 0:
            larger_values = dim_values[larger_indices]
            sorted_larger = larger_indices[np.argsort(-larger_values)]
            remove_larger = sorted_larger[:min(f, len(sorted_larger))]
        else:
            remove_larger = np.array([], dtype=int)

        if len(smaller_indices) > 0:
            smaller_values = dim_values[smaller_indices]
            sorted_smaller = smaller_indices[np.argsort(smaller_values)]
            remove_smaller = sorted_smaller[:min(f, len(sorted_smaller))]
        else:
            remove_smaller = np.array([], dtype=int)

        remove_indices = np.concatenate([remove_larger, remove_smaller])
        keep_indices = np.setdiff1d(np.arange(n_neighbors), remove_indices)

        if len(keep_indices) > 0:
            filtered_values.append(neighbor_vhats[keep_indices, dim])
        else:
            filtered_values.append(np.array([]))

    return filtered_values

print("\n" + "=" * 60)
print("测试3: RCP-f 过滤器")
print("=" * 60)

# 测试场景：一个拜占庭节点发送恶意值
v_hat_current = np.array([1.0, 0.0])
neighbor_values = [
    np.array([0.9, 0.1]),   # 正常邻居1
    np.array([1.1, -0.1]),  # 正常邻居2
    np.array([0.95, 0.05]), # 正常邻居3
    np.array([100.0, 50.0]) # 拜占庭邻居（恶意值）
]

print(f"当前值: {v_hat_current}")
print(f"邻居值:")
for i, v in enumerate(neighbor_values):
    print(f"  邻居{i}: {v}")

filtered = apply_rcpf_filter(v_hat_current, neighbor_values, f=1)

print(f"\n过滤后（f=1）:")
for dim in range(2):
    print(f"  维度{dim}: {filtered[dim]}")
    if len(filtered[dim]) > 0:
        print(f"    均值: {np.mean(filtered[dim]):.4f}")

print("\n" + "=" * 60)
print("所有测试完成！")
print("=" * 60)
