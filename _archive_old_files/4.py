import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_sylvester, solve_continuous_are

# ================== 系统参数 ==================
num_agents = 4           # 总节点数调整为4
faulty_agent = 3         # 拜占庭节点索引改为3（0-based）
f = 1                    # 最大容忍拜占庭节点数

# 物理参数（每个倒立摆不同）
m = [0.1 + 0.01 * i for i in range(num_agents)]  # 摆的质量
M = [1.0 + 0.1 * i for i in range(num_agents)]   # 小车质量
l = [0.1 + 0.005 * i for i in range(num_agents)]  # 摆长
P1, P2 = 2, 10

# ================== 通信拓扑 ==================
adj_matrix = np.ones((num_agents, num_agents), dtype=int) - np.eye(num_agents, dtype=int)  # 全连接拓扑

# ================== 代理类定义 ==================
class Agent:
    def __init__(self, index):
        self.index = index
        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, 9.8, 0],
            [0, 0, 0, 1],
            [0, -0.15/(l[index]*M[index]), (M[index]+m[index])*9.8/(l[index]*M[index]), -0.15/M[index]]
        ])
        self.B = np.array([0, 0, 0, 1/(l[index]*M[index])]).reshape(-1, 1)
        
        # 解决Sylvester方程
        S = np.array([[0, 1], [-1, 0]])
        Ei = np.array([[0, 0], [P1, 0], [0, 0], [P2, 0]])
        Xi = solve_sylvester(self.A, -S, -Ei)
        Ui = np.zeros((1, 2))
        Ui[0, 1] = Ei[-1, 1]/self.B[-1, 0]
        
        # 求解Riccati方程
        Q = np.diag([1, 1, 10, 1])
        R = np.array([[0.1]])
        P = solve_continuous_are(self.A, self.B, Q, R)
        self.K11 = (-np.linalg.inv(R) @ self.B.T @ P).flatten()
        self.K12 = Ui - np.dot(self.K11, Xi)

    def dynamics(self, x, v_hat):
        u = np.dot(self.K11, x) + np.dot(self.K12, v_hat)
        return (self.A @ x + self.B @ u).flatten()

# ================== RCP-f 过滤器 ==================
def rcp_filter(values, f):
    if len(values) <= 2*f:
        return values
    sorted_values = sorted(values)
    return sorted_values[f:-f]

# ================== 全局系统定义 ==================
def total_system(t, y):
    states = y.reshape(num_agents, 6)
    dvdt = np.zeros((num_agents, 6))
    S = np.array([[0, 1], [-1, 0]])
    v_real = np.array([np.cos(t), np.sin(t)])  # 真实参考信号
    
    for i in range(num_agents):
        x = states[i, :4]
        v_hat = states[i, 4:6]
        
        # 拜占庭节点行为
        if i == faulty_agent:
            malicious_signal = 0.5 * np.random.randn(2)  # 恶意信号
            neighbors_v = [malicious_signal]  # 欺骗性信号
        else:
            # 获取邻居信息
            neighbors = np.where(adj_matrix[i] == 1)[0]
            neighbors_v = [states[j, 4:6] for j in neighbors]
            
            # 应用RCP-f过滤器
            filtered_v = []
            for dim in range(2):
                dim_values = [v[dim] for v in neighbors_v]
                filtered_dim = rcp_filter(dim_values, f)
                filtered_v.append(np.mean(filtered_dim) if filtered_dim else v_hat[dim])
            filtered_v = np.array(filtered_v)
        
        # 领导节点动态（所有节点都能接收真实信号）
        dv_hat = S @ v_hat + 10*(v_real - v_hat) + 1*filtered_v

        dxdt = agents[i].dynamics(x, v_hat)
        dvdt[i, :4] = dxdt
        dvdt[i, 4:6] = dv_hat
    
    return dvdt.flatten()

# ================== 初始化 ==================
agents = [Agent(i) for i in range(num_agents)]
y0 = np.zeros(num_agents * 6)
for i in range(num_agents):
    y0[i*6] = 0.1 + 0.01*i    # 初始位置
    y0[i*6+2] = 0.05          # 初始角度
    y0[i*6+4:i*6+6] = [1.0, 0.0]  # 初始估计值

# ================== 仿真求解 ==================
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)
sol = solve_ivp(total_system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)

# ================== 可视化 ==================
plt.figure(figsize=(16, 10))
colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

# 1. 位置跟踪性能
plt.subplot(2, 2, 1)
for i in range(num_agents):
    style = '--' if i == faulty_agent else '-'
    label = f'Agent {i} (Faulty)' if i == faulty_agent else f'Agent {i}'
    plt.plot(sol.t, sol.y[i*6], 
             linestyle=style, 
             color=colors[i],
             linewidth=1.5,
             alpha=0.8,
             label=label)
plt.plot(sol.t, np.cos(sol.t), 'k-', linewidth=2, label='Reference')
plt.title('Position Tracking Performance')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True, alpha=0.3)
plt.legend(ncol=2)

# 2. 角度跟踪
plt.subplot(2, 2, 2)
for i in range(num_agents):
    style = '--' if i == faulty_agent else '-'
    plt.plot(sol.t, sol.y[i*6+2], 
             linestyle=style,
             color=colors[i],
             linewidth=1.5,
             alpha=0.8)
plt.title('Pendulum Angles')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid(True, alpha=0.3)

# 3. 估计误差
plt.subplot(2, 2, 3)
for i in range(num_agents):
    if i != faulty_agent:
        error = np.linalg.norm(sol.y[i*6+4:i*6+6] - np.array([np.cos(sol.t), np.sin(sol.t)]), axis=0)
        plt.plot(sol.t, error, 
                 color=colors[i],
                 linewidth=1.5,
                 alpha=0.8,
                 label=f'Agent {i}')
plt.title('Estimation Error Norm')
plt.xlabel('Time (s)')
plt.ylabel('Error Norm')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()

# 4. 控制输入对比
plt.subplot(2, 2, 4)
for i in range(num_agents):
    u_history = [agents[i].K11 @ y[i*6:i*6+4] + agents[i].K12 @ y[i*6+4:i*6+6] 
                 for y in sol.y.T]
    plt.plot(sol.t, u_history,
             color=colors[i],
             linewidth=1.5,
             alpha=0.6)
plt.title('Control Input Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Control Force (N)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()