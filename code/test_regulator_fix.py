#!/usr/bin/env python3
"""
测试新的调节方程求解方法
"""
import numpy as np

# 系统参数
index = 0
mi = 0.1 * (index + 1)
Mi = 1.0 * (index + 1)
li = 0.1 * (index + 1)
g = 9.8
friction = 0.15

# 系统矩阵
mu_i1 = friction / (li * Mi)
mu_i2 = (Mi + mi) * g / (li * Mi)
mu_i3 = -friction / Mi
bi = 1.0 / (li * Mi)

A = np.array([
    [0, 1, 0, 0],
    [0, 0, g, 0],
    [0, 0, 0, 1],
    [0, mu_i1, mu_i2, mu_i3]
])

B = np.array([[0], [0], [0], [bi]])

pi1 = 2.0 / Mi
pi2 = 1.0 / (li * Mi)
E = np.array([[0, 0], [pi1, 0], [0, 0], [pi2, 0]])

C = np.array([[1, 0, -li, 0]])
F = np.array([[-1, 0]])

S = np.array([[0, 1], [-1, 0]])

print("=" * 70)
print(f"Agent {index} 参数:")
print(f"mi={mi}, Mi={Mi}, li={li}")
print("=" * 70)

# 方法1：参数化求解
# 从输出约束 C@Xi + F = 0 得到:
# Xi[0,:] - li*Xi[2,:] = [1, 0]
# 设 Xi = [[xi1], [xi2], [xi3], [xi4]]，每行2维

# 参数化: Xi[2,:] = [a, b]（自由变量）
#         Xi[0,:] = [1, 0] + li*[a, b] = [1+li*a, li*b]
#         Xi[1,:] = [c, d]（自由变量）
#         Xi[3,:] = [e, f]（自由变量）

# 从状态方程 A@Xi + B@Ui + E = Xi@S 求解其余变量
# 这是一个8个方程（4x2矩阵）8个未知数（a,b,c,d,e,f + Ui的2个元素）

# 展开状态方程:
# 第0行: [0,1,0,0]@Xi = c*[1,0] + d*[0,1] = [c, d]
#        右侧: Xi[0,:]@S = [1+li*a, li*b]@[[0,1],[-1,0]] = [-li*b, 1+li*a]
#        所以: [c, d] = [-li*b, 1+li*a]

# 第1行: [0,0,g,0]@Xi = g*Xi[2,:] = g*[a, b]
#        右侧: Xi[1,:]@S + E[1,:] = [c,d]@S + [pi1,0] = [-d, c] + [pi1, 0]
#        所以: g*[a, b] = [-d + pi1, c]

# 第2行: [0,0,0,1]@Xi = Xi[3,:] = [e, f]
#        右侧: Xi[2,:]@S = [a,b]@S = [-b, a]
#        所以: [e, f] = [-b, a]

# 第3行: [0, mu_i1, mu_i2, mu_i3]@Xi + B@Ui + E[3,:] = Xi[3,:]@S
#        左侧: mu_i1*Xi[1,:] + mu_i2*Xi[2,:] + mu_i3*Xi[3,:] + bi*Ui + [pi2, 0]
#        右侧: [e,f]@S = [-f, e]

print("\n使用参数化方法求解:")
print("-" * 70)

# 从上面的推导:
# c = -li*b,  d = 1+li*a  (从第0行)
# g*a = -d + pi1 = -(1+li*a) + pi1  => g*a + li*a = pi1 - 1  => a*(g+li) = pi1-1
# g*b = c = -li*b  => g*b + li*b = 0  => b*(g+li) = 0  => b = 0

a = (pi1 - 1) / (g + li)
b = 0
c = -li * b
d = 1 + li * a
e = -b
f = a

Xi_param = np.array([
    [1 + li*a, li*b],
    [c, d],
    [a, b],
    [e, f]
])

print(f"Xi (参数化):\n{Xi_param}")

# 验证输出方程
output_residual = C @ Xi_param + F
print(f"\n输出方程 C@Xi + F = 0:")
print(f"残差: {output_residual}")
print(f"残差范数: {np.linalg.norm(output_residual):.6e}")

# 从第3行求解Ui
# mu_i1*[c,d] + mu_i2*[a,b] + mu_i3*[e,f] + bi*Ui + [pi2,0] = [-f, e]
lhs_without_u = mu_i1*np.array([c,d]) + mu_i2*np.array([a,b]) + mu_i3*np.array([e,f]) + np.array([pi2, 0])
rhs = np.array([-f, e])
Ui_vec = (rhs - lhs_without_u) / bi
Ui_param = Ui_vec.reshape(1, 2)

print(f"\nUi (参数化):\n{Ui_param}")

# 验证状态方程
state_residual = A @ Xi_param + B @ Ui_param + E - Xi_param @ S
print(f"\n状态方程 A@Xi + B@Ui + E = Xi@S:")
print(f"残差:\n{state_residual}")
print(f"残差范数: {np.linalg.norm(state_residual):.6e}")

# 测试稳态
print("\n" + "=" * 70)
print("测试稳态跟踪:")
print("=" * 70)

t_test = 5.0
v_test = np.array([np.cos(t_test), np.sin(t_test)])
print(f"\n时间 t={t_test}, v(t) = {v_test}")

x_ss = Xi_param @ v_test
u_ss = Ui_param @ v_test
e_ss = C @ x_ss + F @ v_test

print(f"期望稳态 x_ss = Xi@v: {x_ss}")
print(f"期望控制 u_ss = Ui@v: {u_ss}")
print(f"输出误差 e_ss = C@x_ss + F@v: {e_ss}")
print(f"输出误差范数: {np.linalg.norm(e_ss):.6e}")

# 验证稳态确实满足动力学
dx_ss = A @ x_ss + B @ u_ss + E @ v_test
dv = S @ v_test
expected_dx = Xi_param @ dv

print(f"\n稳态动力学验证:")
print(f"dx_ss = A@x_ss + B@u_ss + E@v: {dx_ss}")
print(f"期望 dx_ss = Xi@(S@v): {expected_dx}")
print(f"差异: {np.linalg.norm(dx_ss - expected_dx):.6e}")
