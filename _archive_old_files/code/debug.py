import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from code import total_system, num_agents, faulty_agent, adj_matrix, agents

def debug_total_system():
    t_span = [0, 10]  # 运行 10s 进行测试
    y0 = np.random.rand(num_agents * 6)  # 随机初始化状态
    
    sol = solve_ivp(total_system, t_span, y0, method='RK45', t_eval=np.linspace(0, 10, 100))
    
    for i in range(num_agents):
        print(f"Agent {i}: Final state = {sol.y[:, -1][i*6:(i+1)*6]}")

    # 误差计算
    error_norms = []
    for i in range(num_agents):
        if i == faulty_agent:
            # 计算拜占庭节点的干扰信号
            interference_signal = sol.y[i*6+4:(i+1)*6, :]  # 取 v_hat
            error_norms.append(np.linalg.norm(interference_signal, axis=0))  # 计算干扰信号范数
        else:
            # 正常节点仍然计算 v_hat 和 v_real 之差
            v_hat_traj = sol.y[i*6+4:(i+1)*6, :]
            v_real_traj = np.array([np.cos(sol.t), np.sin(sol.t)])
            error_norms.append(np.linalg.norm(v_hat_traj - v_real_traj, axis=0))
    
    # 绘图
    plt.figure()
    for i in range(num_agents):
        plt.plot(sol.t, error_norms[i], label=f'Agent {i}')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Error Norm' if faulty_agent is None else 'Error/Interference Norm')
    plt.title('Estimation Error & Byzantine Interference Debugging')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    debug_total_system()
