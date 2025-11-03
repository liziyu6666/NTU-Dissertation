import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_sylvester
from scipy.signal import place_poles

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
        ])  # Ai 计算方式

        self.B = np.array([[0], [0], [0], [1 / (l[index] * M[index])]])  # 控制输入矩阵 Bi
        self.E = np.array([[0, 0], [P1, 0], [0, 0], [P2, 0]])  # Ei 决定参考信号 v(t) 如何影响系统
        self.C = np.array([[1, 0, -l[index], 0]])
        self.F = np.array([[-1, 0]])

        self.Xi = solve_sylvester(self.A, -S, -self.E)  # Ai Xi + Xi S + E = 0
        self.Ui = np.linalg.lstsq(self.B, -self.Xi @ S - self.E, rcond=None)[0]

        desired_poles = np.array([-2, -2.5, -3, -3.5])  # Hurwitz 极点
        place_result = place_poles(self.A, self.B, desired_poles)
        self.K11 = place_result.gain_matrix.flatten()

        self.K12 = self.Ui - np.dot(self.K11, self.Xi)

    def dynamics(self, x, v_hat):
        u = np.dot(self.K11, x) + np.dot(self.K12, v_hat)
        return (self.A @ x + self.B @ u).flatten()


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
        neighbor_vhats = [states[j, 4:6] for j in neighbors]

        if len(neighbor_vhats) > 2 * f:
            sorted_vhats = sorted(neighbor_vhats, key=lambda v: np.linalg.norm(v))
            robust_vhat_neighbors = sorted_vhats[f: -f]
        else:
            robust_vhat_neighbors = neighbor_vhats

        if len(robust_vhat_neighbors) > 0:
            robust_v_hat = np.mean(robust_vhat_neighbors, axis=0)
        else:
            robust_v_hat = v_hat

        # 记录 v_hat 与邻居均值的差异
        diff_norm = np.linalg.norm(v_hat - robust_v_hat)
        current_diff.append(diff_norm)

        if i == faulty_agent:
            dv_hat = np.array([50 * np.sin(10 * t) + 15 * np.cos(12 * t), t / 15])
        else:
            error_sum = sum(robust_v_hat - v_hat for j in neighbors)
            dv_hat = S @ v_hat + 10 * error_sum

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
t_span = (0, 5)
t_eval = np.linspace(*t_span, 1000)
sol = solve_ivp(total_system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)

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

# 1. 位置跟踪
plt.subplot(2, 3, 1)
for i in range(num_agents):
    plt.plot(sol.t, sol.y[i * 6], linestyle='--' if i == faulty_agent else '-', color=colors[i])
plt.plot(sol.t, np.cos(sol.t), 'k-', linewidth=2, label='Reference')
plt.title('Position Tracking')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
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
    plt.plot(sol.t, estimation_error[i], color=colors[i])
plt.title('Estimation Error Norm')
plt.xlabel('Time (s)')
plt.ylabel('Error Norm')
plt.yscale('log')
plt.grid(True)

# 4. 控制输入
plt.subplot(2, 3, 4)
for i in range(num_agents):
    u_history = [agents[i].K11 @ y[i * 6:i * 6 + 4] + agents[i].K12 @ y[i * 6 + 4:i * 6 + 6] for y in sol.y.T]
    plt.plot(sol.t, u_history, color=colors[i])
plt.title('Control Input Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Control Force (N)')
plt.grid(True)

# 5. 位置误差指标
plt.subplot(2, 3, 5)
for i in range(num_agents):
    plt.plot(sol.t, position_error[i], label=f'Agent {i}', color=colors[i])
plt.title('Position Error')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.grid(True)

# 6. v_hat 差异监控（观测器收敛性分析）
plt.subplot(2, 3, 6)
vhat_diff_log_np = np.array(vhat_diff_log)
for i in range(num_agents):
    plt.plot(vhat_diff_time_log, vhat_diff_log_np[:, i], color=colors[i], label=f'Agent {i}')
plt.title('v_hat vs. Robust Neighbor Mean')
plt.xlabel('Time (s)')
plt.ylabel('|| v_hat - mean ||')
plt.yscale('log')
plt.grid(True)

plt.tight_layout()
plt.show()

# ================== 保存 v_hat 差值日志为 CSV 文件 ==================
csv_filename = "vhat_difference_log.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['time'] + [f'agent{i}' for i in range(num_agents)]
    writer.writerow(header)
    
    for t, row in zip(vhat_diff_time_log, vhat_diff_log):
        writer.writerow([t] + list(row))

print(f"✅ v_hat 差值日志已保存至 {csv_filename}")
