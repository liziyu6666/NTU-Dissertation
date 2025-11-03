"""
调试控制器设计
"""
import numpy as np
from scipy.signal import place_poles

# 复制Agent类的初始化
index = 0  # 测试Agent 0

# 物理参数
m = [0.1 * (i + 1) for i in range(8)]
M = [1.0 * (i + 1) for i in range(8)]
l = [0.1 * (i + 1) for i in range(8)]
g = 9.8
friction = 0.15

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

E = np.array([[0, 0], [2.0/Mi, 0], [0, 0], [1.0/(li*Mi), 0]])

C = np.array([[1, 0, -li, 0]])
F = np.array([[-1, 0]])

S = np.array([[0, 1], [-1, 0]])

print("="*70)
print("Controller Design Debug for Agent 0")
print("="*70)

print("\nSystem Matrices:")
print(f"A = \n{A}")
print(f"\nB = \n{B}")
print(f"\nE = \n{E}")

# 检查系统的特征值（开环）
eigenvalues_open = np.linalg.eigvals(A)
print(f"\nOpen-loop eigenvalues: {eigenvalues_open}")
print(f"System is unstable (has positive real parts): {np.any(eigenvalues_open.real > 0)}")

# 检查可控性
ctrl_matrix = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
rank = np.linalg.matrix_rank(ctrl_matrix)
print(f"\nControllability matrix rank: {rank}/4")
print(f"System is controllable: {rank == 4}")

# 设计控制器
desired_poles = np.array([-2, -2.5, -3, -3.5])
print(f"\nDesired poles: {desired_poles}")

try:
    place_result = place_poles(A, B, desired_poles)
    K11 = place_result.gain_matrix.flatten()
    print(f"K11 (feedback gain): {K11}")

    # 检查闭环特征值
    A_closed = A + B @ K11.reshape(1, -1)
    eigenvalues_closed = np.linalg.eigvals(A_closed)
    print(f"\nClosed-loop eigenvalues: {eigenvalues_closed}")
    print(f"All eigenvalues have negative real part: {np.all(eigenvalues_closed.real < 0)}")

    # 检查增益的大小
    print(f"\nGain magnitudes:")
    for i, gain in enumerate(K11):
        print(f"  K11[{i}] = {gain:12.6f}")

    # 求解调节方程检查Xi和Ui
    q = S.shape[0]
    n = A.shape[0]
    m_ctrl = B.shape[1]
    p = C.shape[0]

    I_n = np.eye(n)
    I_q = np.eye(q)

    A11 = np.kron(S.T, I_n) - np.kron(I_q, A)
    A12 = -np.kron(I_q, B)
    A21 = np.kron(I_q, C)
    A22 = np.zeros((p*q, m_ctrl*q))

    A_top = np.hstack([A11, A12])
    A_bot = np.hstack([A21, A22])
    A_mat = np.vstack([A_top, A_bot])

    b_top = E.flatten('F')
    b_bot = -F.flatten('F')
    b_vec = np.concatenate([b_top, b_bot])

    solution, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)

    Xi = solution[:n*q].reshape((n, q), order='F')
    Ui = solution[n*q:].reshape((m_ctrl, q), order='F')

    print(f"\nRegulator Equations Solution:")
    print(f"Xi = \n{Xi}")
    print(f"Ui = \n{Ui}")

    K12 = Ui - K11.reshape(1, -1) @ Xi
    print(f"\nK12 = \n{K12}")

    # 检查K12的大小
    print(f"\nK12 magnitudes: {np.linalg.norm(K12, axis=0)}")

    # 测试：对于典型的v_hat值，控制输入是多少？
    print(f"\n" + "-"*70)
    print("Test Control Inputs for typical v_hat values:")
    print("-"*70)

    test_x = np.array([0.1, 0.0, 0.05, 0.0])  # 初始状态
    test_vhats = [
        np.array([1.0, 0.0]),  # 初始值
        np.array([np.cos(5), np.sin(5)]),  # t=5
        np.array([np.cos(10), np.sin(10)]),  # t=10
        np.array([np.cos(15), np.sin(15)]),  # t=15
    ]

    for i, v_hat in enumerate(test_vhats):
        u = K11 @ test_x + K12.flatten() @ v_hat
        dx_dt = A @ test_x + B.flatten() * u + E @ v_hat
        print(f"\nv_hat = [{v_hat[0]:7.4f}, {v_hat[1]:7.4f}]")
        print(f"  u = {u:12.6f}")
        print(f"  dx/dt[0] = {dx_dt[0]:12.6e}")
        print(f"  dx/dt[2] = {dx_dt[2]:12.6e}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("="*70)
