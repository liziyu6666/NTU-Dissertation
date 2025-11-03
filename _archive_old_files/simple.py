import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are, solve_sylvester

# ================== 设置支持中文的字体 ==================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统常用字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ================== 系统参数 ==================
m = 0.1   # 摆的质量
M = 1.0   # 小车质量
l = 0.1   # 摆长
P1 = 2
P2 = 10

# ================== 系统矩阵定义 ==================
A = np.array([
    [0, 1, 0, 0],
    [0, 0, 9.8, 0],
    [0, 0, 0, 1],
    [0, -0.15/(l*M), (M+m)*9.8/(l*M), -0.15/M]
])
B = np.array([0, 0, 0, 1/(l*M)]).reshape(-1, 1)  # 转换为列向量

# ================== 外部信号参数 ==================
S = np.array([[0, 1], [-1, 0]])  # 外部信号动态矩阵
Ei = np.array([                  # 外部信号输入矩阵
    [0, 0],
    [P1, 0],
    [0, 0],
    [P2, 0]  # 假设外部信号第二个分量影响角加速度
])

# ================== 理论增益计算 ==================
Xi = solve_sylvester(A, -S, -Ei)  # 解调节器方程 (Ai Xi - Xi S) = -Ei
Ui = np.zeros((1, 2))
Ui[0, 1] = Ei[-1, 1] / B[-1, 0]  # 计算 Ui

# 计算 LQR 控制增益 K11
Q = np.diag([1, 1, 10, 1])  # 状态权重矩阵
R = np.array([[0.1]])        # 控制输入权重
P = solve_continuous_are(A, B, Q, R)
K11 = -np.linalg.inv(R) @ B.T @ P
K11 = K11.flatten()
K12 = Ui - np.dot(K11, Xi)

# ================== 动力学模型 ==================
def dynamics(x, v_hat, t):
    """倒立摆系统的动力学"""
    u = np.dot(K11, x) + np.dot(K12, v_hat)
    return (A @ x + B @ u).flatten()

# ================== 信号估计器 ==================
def total_system(t, y):
    x = y[:4]       # 系统状态: [位置, 速度, 角度, 角速度]
    v_hat = y[4:6]  # 估计的外部信号
    
    v_real = np.array([np.cos(t), np.sin(t)])  # 真实外部信号
    dv_hat = S @ v_hat + 50 * (v_real - v_hat)  # 增强误差修正
    dxdt = dynamics(x, v_hat, t)
    
    dydt = np.zeros(6)
    dydt[:4] = dxdt
    dydt[4:6] = dv_hat
    return dydt

# ================== 仿真设置 ==================
y0 = np.zeros(6)
y0[0] = 0.1    # 初始位置
y0[2] = 0.05   # 初始角度
y0[4:6] = [1, 0]  # 初始信号估计

t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)

# ================== 求解ODE ==================
sol = solve_ivp(total_system, t_span, y0, t_eval=t_eval, 
                method='RK45', rtol=1e-5, atol=1e-8)

# ================== 可视化结果 ==================
plt.figure(figsize=(12, 8))

# 1. 位置跟踪
plt.subplot(2, 2, 1)
plt.plot(sol.t, sol.y[0], label='实际位置')
plt.plot(sol.t, np.cos(sol.t), 'r--', label='参考位置')
plt.title('位置跟踪性能')
plt.xlabel('时间 (s)')
plt.ylabel('位置 (m)')
plt.legend()

# 2. v_hat 与 v(t) 对比
plt.subplot(2, 2, 2)
plt.plot(sol.t, sol.y[4], label='v_hat_1')
plt.plot(sol.t, np.cos(sol.t), 'r--', label='真实 v1')
plt.plot(sol.t, sol.y[5], label='v_hat_2')
plt.plot(sol.t, np.sin(sol.t), 'g--', label='真实 v2')
plt.title('外部信号估计对比')
plt.xlabel('时间 (s)')
plt.ylabel('信号估计值')
plt.legend()

# 3. 信号估计误差
plt.subplot(2, 2, 3)
error = np.linalg.norm(sol.y[4:6].T - np.vstack([np.cos(sol.t), np.sin(sol.t)]).T, axis=1)
plt.plot(sol.t, error, 'm-')
plt.title('信号估计误差')
plt.xlabel('时间 (s)')
plt.ylabel('误差范数')

# 4. 控制输入
u_history = [np.dot(K11, y[:4]) + np.dot(K12, y[4:6]) for y in sol.y.T]
plt.subplot(2, 2, 4)
plt.plot(sol.t, u_history, 'b-')
plt.title('控制输入')
plt.xlabel('时间 (s)')
plt.ylabel('控制力 (N)')

plt.tight_layout()
plt.show()
