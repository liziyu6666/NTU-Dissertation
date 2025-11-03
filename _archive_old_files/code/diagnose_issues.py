#!/usr/bin/env python3
"""
诊断脚本：检查系统稳定性和调节方程
"""
import numpy as np
from scipy.linalg import solve_sylvester, solve_continuous_are
from scipy.signal import place_poles

# 系统参数
num_agents = 8
m = [0.1 * (i + 1) for i in range(num_agents)]
M = [1.0 * (i + 1) for i in range(num_agents)]
l = [0.1 * (i + 1) for i in range(num_agents)]
g = 9.8
friction = 0.15

S = np.array([[0, 1], [-1, 0]])

print("=" * 70)
print("诊断1: 检查每个智能体的系统矩阵和稳定性")
print("=" * 70)

for index in range(num_agents):
    print(f"\n{'='*70}")
    print(f"Agent {index}")
    print(f"{'='*70}")

    mi = m[index]
    Mi = M[index]
    li = l[index]
    fi = friction

    # 系统矩阵
    mu_i1 = fi / (li * Mi)
    mu_i2 = (Mi + mi) * g / (li * Mi)
    mu_i3 = -fi / Mi
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

    # 检查开环稳定性
    eigenvalues_A = np.linalg.eigvals(A)
    print(f"开环极点: {eigenvalues_A}")
    print(f"开环稳定性: {'不稳定（有正实部极点）' if np.any(np.real(eigenvalues_A) > 0) else '稳定'}")

    # 检查可控性
    ctrl_matrix = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
    rank = np.linalg.matrix_rank(ctrl_matrix)
    print(f"可控性: rank = {rank}/4, {'可控' if rank == 4 else '不可控'}")

    # 检查调节方程
    try:
        Xi = solve_sylvester(A, -S, -E)
        print(f"\n调节方程 Ai*Xi + Xi*S + Ei = 0:")
        residual = A @ Xi + Xi @ S + E
        print(f"  残差范数: {np.linalg.norm(residual):.6e}")

        # 验证 Ci*Xi + Fi = 0
        output_residual = C @ Xi + F
        print(f"  输出方程 Ci*Xi + Fi = 0:")
        print(f"  残差: {output_residual}")
        print(f"  残差范数: {np.linalg.norm(output_residual):.6e}")

        # 从调节方程求解 Ui
        # 正确的调节方程: Xi*S = Ai*Xi + Ei + Bi*Ui
        # 因此: Bi*Ui = Xi*S - Ai*Xi - Ei
        target = Xi @ S - A @ Xi - E
        Ui = np.linalg.lstsq(B, target, rcond=None)[0]
        print(f"\n控制增益 Ui:")
        print(f"  形状: {Ui.shape}")
        print(f"  值:\n{Ui}")

        # 验证调节方程完整性
        verification = A @ Xi + B @ Ui + E - Xi @ S
        print(f"  完整调节方程验证 (Ai*Xi + Bi*Ui + Ei - Xi*S):")
        print(f"  残差范数: {np.linalg.norm(verification):.6e}")

    except Exception as e:
        print(f"调节方程求解失败: {e}")
        Xi = np.zeros((4, 2))
        Ui = np.zeros((1, 2))

    # 设计反馈增益使闭环稳定
    if rank == 4:
        # 使用LQR设计
        Q = np.diag([100, 10, 100, 10])  # 惩罚位置和角度
        R = np.array([[0.1]])

        try:
            P = solve_continuous_are(A, B, Q, R)
            K11 = (np.linalg.inv(R) @ B.T @ P).flatten()
            K11 = -K11  # 负反馈

            print(f"\nLQR反馈增益 K11: {K11}")

            # 检查闭环稳定性
            A_cl = A + B @ K11.reshape(1, -1)
            eigenvalues_cl = np.linalg.eigvals(A_cl)
            print(f"闭环极点: {eigenvalues_cl}")
            max_real = np.max(np.real(eigenvalues_cl))
            print(f"最大实部: {max_real:.6f}")
            print(f"闭环稳定性: {'稳定' if max_real < -0.01 else '不稳定或边界'}")

            # 计算 K12
            K12 = Ui - K11.reshape(1, -1) @ Xi
            print(f"\nK12 = Ui - K11*Xi:")
            print(f"  形状: {K12.shape}")
            print(f"  值:\n{K12}")

        except Exception as e:
            print(f"LQR设计失败: {e}")
            import traceback
            traceback.print_exc()

print("\n" + "=" * 70)
print("诊断2: 测试参考信号跟踪")
print("=" * 70)

# 测试一个时间点
t_test = 5.0
v_test = np.array([np.cos(t_test), np.sin(t_test)])
print(f"\n时间 t={t_test}s")
print(f"参考信号 v(t) = {v_test}")
print(f"期望稳态: xi_ss = Xi*v(t)")

for index in [0, 4, 7]:  # 测试几个代表性的智能体
    print(f"\nAgent {index}:")

    mi = m[index]
    Mi = M[index]
    li = l[index]
    fi = friction

    mu_i1 = fi / (li * Mi)
    mu_i2 = (Mi + mi) * g / (li * Mi)
    mu_i3 = -fi / Mi
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

    Xi = solve_sylvester(A, -S, -E)
    target = Xi @ S - A @ Xi - E
    Ui = np.linalg.lstsq(B, target, rcond=None)[0]

    # 期望的稳态
    x_ss = Xi @ v_test
    u_ss = Ui @ v_test
    e_ss = C @ x_ss + F @ v_test

    print(f"  期望稳态 x_ss = Xi*v: {x_ss}")
    print(f"  期望控制 u_ss = Ui*v: {u_ss}")
    print(f"  期望输出误差 e_ss = Ci*x_ss + Fi*v: {e_ss}")
    print(f"  输出误差范数: {np.linalg.norm(e_ss):.6e}")

print("\n" + "=" * 70)
print("诊断完成！")
print("=" * 70)
