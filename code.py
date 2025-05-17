import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_sylvester, solve_continuous_are

# ================== 系统参数 ==================
num_agents = 8
faulty_agent = 4
f = 1  # 最大容忍拜占庭节点数

# 物理参数（每个智能体不同）
m = [0.1 + 0.01 * i for i in range(num_agents)]
M = [1.0 + 0.1 * i for i in range(num_agents)]
l = [0.1 + 0.005 * i for i in range(num_agents)]
P1, P2 = 2, 10

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
        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, 9.8 + 0.5 * index, 0],  # 每个智能体的重力项不同
            [0, 0, 0, 1],
            [0, -0.15 / (l[index] * M[index]), 
             (M[index] + m[index]) * (9.8 + 0.5 * index) / (l[index] * M[index]), 
             -0.15 / M[index]]
        ]) # Ai 计算方式
        self.B = np.array([[0], [0], [0], [1 / (l[index] * M[index])]]) # 控制输入矩阵Bi
        self.E = np.array([[0, 0], [P1, 0], [0, 0], [P2, 0]]) # Ei决定参考信号v(t)如何影响系统
        self.C = np.array([[1, 0, -l[index], 0]])
        self.F = np.array([[-1, 0]])
        # 并没有计算ei

        
        # 计算 Sylvester 方程解
        self.Xi = solve_sylvester(self.A, -S, -self.E)    # AiXi + XiS + E = 0;
        self.Ui = np.zeros((1, 2))
        self.Ui[0, 1] = self.E[-1, 1] / self.B[-1, 0]

        # LQR计算K11和K12过程和论文不一样
        # 计算 LQR 增益矩阵
        Q = np.diag([1, 1, 10, 1])
        R = np.array([[0.1]])
        P = solve_continuous_are(self.A, self.B, Q, R)
        self.K11 = (-np.linalg.inv(R) @ self.B.T @ P).flatten()
        self.K12 = self.Ui - np.dot(self.K11, self.Xi)

    def dynamics(self, x, v_hat):
        u = np.dot(self.K11, x) + np.dot(self.K12, v_hat)
        return (self.A @ x + self.B @ u).flatten()

# ================== 计算总系统动态 ==================
def total_system(t, y):
    states = y.reshape(num_agents, 6)
    dvdt = np.zeros((num_agents, 6))
    v_real = np.array([np.cos(t), np.sin(t)])  # 参考信号

    # 这里是如何区分代理和非代理节点并过滤拜占庭节点的没有体现
    for i in range(num_agents):
        x = states[i, :4]
        v_hat = states[i, 4:6]

        if i == faulty_agent:
            dv_hat = np.array([50 * np.sin(10 * t) + 15 * np.cos(12 * t), t / 15])
        else:
            #鲁棒调节器公式后半部分不一致
            dv_hat = S @ v_hat + 10 * (v_real - v_hat)

        dxdt = agents[i].dynamics(x, v_hat)
        dvdt[i, :4] = dxdt
        dvdt[i, 4:6] = dv_hat

    return dvdt.flatten()

# ================== 初始化智能体 ==================
agents = [Agent(i) for i in range(num_agents)]
y0 = np.zeros(num_agents * 6)
for i in range(num_agents):
    y0[i * 6] = 0.1 + 0.01 * i
    y0[i * 6 + 2] = 0.05
    y0[i * 6 + 4:i * 6 + 6] = [1.0, 0.0]

# ================== 运行仿真 ==================
t_span = (0, 40)
t_eval = np.linspace(*t_span, 1000)
sol = solve_ivp(total_system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)

# ================== 可视化 ==================
plt.figure(figsize=(16, 10))
colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

# 1. 位置跟踪
plt.subplot(2, 2, 1)
for i in range(num_agents):
    plt.plot(sol.t, sol.y[i * 6], linestyle='--' if i == faulty_agent else '-', color=colors[i])
plt.plot(sol.t, np.cos(sol.t), 'k-', linewidth=2, label='Reference')
plt.title('Position Tracking')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)

# 2. 角度跟踪
plt.subplot(2, 2, 2)
for i in range(num_agents):
    plt.plot(sol.t, sol.y[i * 6 + 2], linestyle='--' if i == faulty_agent else '-', color=colors[i])
plt.title('Pendulum Angles')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid(True)

# 3. 估计误差
plt.subplot(2, 2, 3)
for i in range(num_agents):
    error = np.linalg.norm(sol.y[i * 6 + 4:i * 6 + 6] - np.array([np.cos(sol.t), np.sin(sol.t)]), axis=0)
    plt.plot(sol.t, error, color=colors[i])
plt.title('Estimation Error Norm')
plt.xlabel('Time (s)')
plt.ylabel('Error Norm')
plt.yscale('log')
plt.grid(True)

# 4. 控制输入对比
plt.subplot(2, 2, 4)
for i in range(num_agents):
    u_history = [agents[i].K11 @ y[i * 6:i * 6 + 4] + agents[i].K12 @ y[i * 6 + 4:i * 6 + 6] for y in sol.y.T]
    plt.plot(sol.t, u_history, color=colors[i])
plt.title('Control Input Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Control Force (N)')
plt.grid(True)

plt.show()
