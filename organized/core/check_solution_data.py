"""
直接检查仿真解的数值数据
"""
import numpy as np
from scipy.integrate import solve_ivp
import sys

# 复制1.py中的关键部分
num_agents = 8
faulty_agent = 4
f = 1

# 简化版：只检查解的数值
print("Checking solution data...")
print("="*70)

# 从1.py导入
sys.path.insert(0, '/home/liziyu/d/dissertation/organized/core')
import importlib.util
spec = importlib.util.spec_from_file_location("sim", "1.py")
sim_module = importlib.util.module_from_spec(spec)

# 重新运行仿真，直接捕获解
print("\nRunning simulation with FIXED Byzantine attack...")
print("-"*70)

exec(open('1.py').read())

# 检查解的数值
print("\nAnalyzing solution...")
print("-"*70)

# 检查最后时刻的状态
t_final = sol.t[-1]
y_final = sol.y[:, -1]

print(f"\nTime at last step: {t_final:.6f} seconds")
print(f"Expected: 15.000000 seconds")

print(f"\nStates at t={t_final:.2f}:")
for i in range(num_agents):
    x1 = y_final[i * 6]      # position
    x2 = y_final[i * 6 + 1]  # velocity
    x3 = y_final[i * 6 + 2]  # angle
    x4 = y_final[i * 6 + 3]  # angular velocity
    v_hat_0 = y_final[i * 6 + 4]
    v_hat_1 = y_final[i * 6 + 5]

    print(f"\nAgent {i}{'(Byzantine)' if i == faulty_agent else ''}:")
    print(f"  x1 (position):     {x1:15.6e}")
    print(f"  x3 (angle):        {x3:15.6e}")
    print(f"  v_hat: [{v_hat_0:10.6f}, {v_hat_1:10.6f}]")

    if abs(x1) > 1e10 or abs(x3) > 1e10:
        print(f"  ⚠️  DIVERGED!")

# 检查中间时刻
print(f"\n" + "="*70)
print("Checking intermediate time steps...")
print("-"*70)

check_times = [5.0, 10.0, 12.0, 13.0, 14.0, 14.5, 14.9, 15.0]
for t_check in check_times:
    idx = np.argmin(np.abs(sol.t - t_check))
    t_actual = sol.t[idx]

    # Agent 0 (normal)
    x1_0 = sol.y[0, idx]
    x3_0 = sol.y[2, idx]

    # Agent 4 (Byzantine)
    x1_4 = sol.y[4*6, idx]
    x3_4 = sol.y[4*6 + 2, idx]

    diverged_0 = abs(x1_0) > 1e10 or abs(x3_0) > 1e10
    diverged_4 = abs(x1_4) > 1e10 or abs(x3_4) > 1e10

    status_0 = "DIVERGED" if diverged_0 else "OK"
    status_4 = "DIVERGED" if diverged_4 else "OK"

    print(f"t={t_actual:6.3f}: Agent0 x1={x1_0:12.4e} [{status_0:8}], " +
          f"Agent4 x1={x1_4:12.4e} [{status_4:8}]")

print("="*70)
