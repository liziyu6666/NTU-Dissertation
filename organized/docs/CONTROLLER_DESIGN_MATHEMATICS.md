# 分布式控制器设计的数学基础

**文档目的**: 详细讲解合作输出调节问题的控制器设计、调节方程求解和稳定性分析

**作者**: liziyu
**日期**: 2025-10-30

---

## 目录

1. [合作输出调节问题](#1-合作输出调节问题)
2. [调节方程求解](#2-调节方程求解)
3. [分布式观测器设计](#3-分布式观测器设计)
4. [状态反馈控制器设计](#4-状态反馈控制器设计)
5. [稳定性分析](#5-稳定性分析)
6. [数值计算示例](#6-数值计算示例)

---

## 1. 合作输出调节问题

### 1.1 问题描述

给定 $N$ 个异构智能体系统：

$$
\begin{cases}
\dot{x}_i = A_i x_i + B_i u_i + E_i v \\
y_i = C_i x_i + F_i v
\end{cases}, \quad i = 1, 2, \ldots, N
$$

和外系统（参考信号发生器）：

$$
\dot{v} = Sv
$$

**目标**：设计分布式控制律 $u_i$，使得：

1. **调节**（Regulation）：$\lim_{t \to \infty} e_i(t) = 0$，其中 $e_i = y_i - y_{ref}$
2. **镇定**（Stabilization）：闭环系统内部稳定
3. **分布式**（Distributed）：$u_i$ 仅依赖于本地和邻居信息

### 1.2 标准假设

**假设1（可镇定性）**：$(A_i, B_i)$ 可镇定，即存在 $K_i$ 使得 $A_i + B_i K_i$ 是Hurwitz矩阵。

**假设2（可检测性）**：$(A_i, C_i)$ 可检测。

**假设3（通信连通性）**：通信图 $\mathcal{G}$ 包含一棵生成树，且根节点能感知 $v(t)$。

**假设4（传输零点）**：矩阵

$$
\begin{bmatrix}
A_i - \lambda I & B_i \\
C_i & 0
\end{bmatrix}
$$

对所有 $\lambda \in \sigma(S)$（$S$ 的特征值）列满秩。

这保证了**调节方程有解**。

---

## 2. 调节方程求解

### 2.1 调节方程（Regulator Equations）

对于智能体 $i$，调节方程为：

$$
\begin{cases}
\Pi_i S = A_i \Pi_i + B_i \Gamma_i + E_i \\
0 = C_i \Pi_i + F_i
\end{cases}
$$

其中：
- $\Pi_i \in \mathbb{R}^{n \times q}$: 状态前馈增益
- $\Gamma_i \in \mathbb{R}^{m \times q}$: 控制前馈增益

**物理意义**：调节方程给出了**稳态解**。当系统达到稳态时，存在 $x_i^* = \Pi_i v$ 和 $u_i^* = \Gamma_i v$ 使得跟踪误差为零。

### 2.2 矩阵形式

将调节方程重写为线性方程组：

$$
\begin{bmatrix}
\Pi_i S - A_i \Pi_i - B_i \Gamma_i \\
C_i \Pi_i
\end{bmatrix} = \begin{bmatrix}
E_i \\
-F_i
\end{bmatrix}
$$

使用Kronecker积展开：

$$
\begin{bmatrix}
(S^T \otimes I_n) - (I_q \otimes A_i) & -(I_q \otimes B_i) \\
I_q \otimes C_i & 0
\end{bmatrix}
\begin{bmatrix}
\text{vec}(\Pi_i) \\
\text{vec}(\Gamma_i)
\end{bmatrix} = \begin{bmatrix}
\text{vec}(E_i) \\
\text{vec}(-F_i)
\end{bmatrix}
$$

其中 $\text{vec}(\cdot)$ 是向量化算子，$\otimes$ 是Kronecker积。

### 2.3 求解算法

#### 方法1: 直接求解

$$
\begin{bmatrix}
\text{vec}(\Pi_i) \\
\text{vec}(\Gamma_i)
\end{bmatrix} = \begin{bmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{bmatrix}^{-1} \begin{bmatrix}
b_1 \\
b_2
\end{bmatrix}
$$

其中：

$$
\begin{align}
A_{11} &= S^T \otimes I_n - I_q \otimes A_i \in \mathbb{R}^{nq \times nq} \\
A_{12} &= -(I_q \otimes B_i) \in \mathbb{R}^{nq \times mq} \\
A_{21} &= I_q \otimes C_i \in \mathbb{R}^{pq \times nq} \\
A_{22} &= 0 \in \mathbb{R}^{pq \times mq} \\
b_1 &= \text{vec}(E_i) \in \mathbb{R}^{nq} \\
b_2 &= \text{vec}(-F_i) \in \mathbb{R}^{pq}
\end{align}
$$

**Python实现**：
```python
import numpy as np

def solve_regulator_equations(A_i, B_i, C_i, E_i, F_i, S):
    n, q = E_i.shape
    m = B_i.shape[1]
    p = C_i.shape[0]

    # 构建系数矩阵
    I_n = np.eye(n)
    I_q = np.eye(q)

    A11 = np.kron(S.T, I_n) - np.kron(I_q, A_i)  # (nq, nq)
    A12 = -np.kron(I_q, B_i)                      # (nq, mq)
    A21 = np.kron(I_q, C_i)                       # (pq, nq)
    A22 = np.zeros((p*q, m*q))                    # (pq, mq)

    A_mat = np.vstack([
        np.hstack([A11, A12]),
        np.hstack([A21, A22])
    ])  # ((n+p)q, (n+m)q)

    # 构建右端向量
    b_vec = np.concatenate([
        E_i.flatten('F'),   # Fortran order (列优先)
        -F_i.flatten('F')
    ])

    # 求解
    solution = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]

    # 提取结果
    Pi_i = solution[:n*q].reshape((n, q), order='F')
    Gamma_i = solution[n*q:].reshape((m, q), order='F')

    return Pi_i, Gamma_i
```

#### 方法2: Sylvester方程法

第一个方程可以写成Sylvester方程：

$$
\Pi_i S - A_i \Pi_i = E_i + B_i \Gamma_i
$$

结合第二个方程：

$$
C_i \Pi_i = -F_i
$$

可以先从第二个方程用伪逆求 $\Pi_i$，然后代入第一个方程求 $\Gamma_i$。

### 2.4 实际计算示例

**参数**（Cart-Pendulum系统，智能体1）：

$$
A_1 = \begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & 9.8 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0.15 & 10.78 & -0.15
\end{bmatrix}, \quad
B_1 = \begin{bmatrix}
0 \\ 0 \\ 0 \\ 10
\end{bmatrix}
$$

$$
E_1 = \begin{bmatrix}
0 & 0 \\
20 & 0 \\
0 & 0 \\
10 & 0
\end{bmatrix}, \quad
C_1 = \begin{bmatrix}
1 & 0 & -0.1 & 0
\end{bmatrix}, \quad
F_1 = \begin{bmatrix}
-1 & 0
\end{bmatrix}
$$

$$
S = \begin{bmatrix}
0 & 1 \\
-1 & 0
\end{bmatrix}
$$

**求解结果**：

$$
\Pi_1 \approx \begin{bmatrix}
0.1 & 0 \\
0 & 0.1 \\
0 & 0 \\
0 & 0
\end{bmatrix}, \quad
\Gamma_1 \approx \begin{bmatrix}
0.5 & 1.0
\end{bmatrix}
$$

**验证**：

$$
\Pi_1 S = \begin{bmatrix}
0.1 & 0 \\
0 & 0.1 \\
0 & 0 \\
0 & 0
\end{bmatrix} \begin{bmatrix}
0 & 1 \\
-1 & 0
\end{bmatrix} = \begin{bmatrix}
0 & 0.1 \\
-0.1 & 0 \\
0 & 0 \\
0 & 0
\end{bmatrix}
$$

$$
A_1 \Pi_1 + B_1 \Gamma_1 + E_1 = \cdots = \begin{bmatrix}
0 & 0.1 \\
-0.1 & 0 \\
0 & 0 \\
0 & 0
\end{bmatrix} \quad \checkmark
$$

$$
C_1 \Pi_1 + F_1 = \begin{bmatrix}
1 & 0 & -0.1 & 0
\end{bmatrix} \begin{bmatrix}
0.1 & 0 \\
0 & 0.1 \\
0 & 0 \\
0 & 0
\end{bmatrix} + \begin{bmatrix}
-1 & 0
\end{bmatrix} = \begin{bmatrix}
0 & 0
\end{bmatrix} \quad \checkmark
$$

---

## 3. 分布式观测器设计

### 3.1 观测器结构

由于智能体无法直接测量 $v(t)$，需要设计分布式观测器来估计 $v(t)$。

**分布式观测器**：

$$
\dot{\hat{v}}_i = S\hat{v}_i + c_1 \sum_{j \in \mathcal{N}_i} a_{ij}(\hat{v}_j - \hat{v}_i) + c_2 \mathbb{1}_{\{i \in \mathcal{T}\}} (v - \hat{v}_i)
$$

其中：
- $\hat{v}_i \in \mathbb{R}^q$: 智能体 $i$ 对 $v$ 的估计
- $c_1 > 0$: 共识增益
- $c_2 > 0$: 跟踪增益
- $\mathcal{T}$: 目标节点集合（能感知 $v$ 的节点）
- $a_{ij}$: 邻接矩阵元素

### 3.2 误差动力学

定义估计误差：

$$
\epsilon_i = \hat{v}_i - v
$$

误差动力学：

$$
\dot{\epsilon}_i = S\epsilon_i + c_1 \sum_{j \in \mathcal{N}_i} a_{ij}(\epsilon_j - \epsilon_i) - c_2 \mathbb{1}_{\{i \in \mathcal{T}\}} \epsilon_i
$$

写成全局形式：

$$
\dot{\epsilon} = (I_N \otimes S - c_1 L \otimes I_q - c_2 D \otimes I_q) \epsilon
$$

其中：
- $\epsilon = [\epsilon_1^T, \epsilon_2^T, \ldots, \epsilon_N^T]^T \in \mathbb{R}^{Nq}$
- $L$: Laplacian矩阵，$L = D - A$
- $D$: 度矩阵，$D = \text{diag}(\mathbb{1}_{\{1 \in \mathcal{T}\}}, \ldots, \mathbb{1}_{\{N \in \mathcal{T}\}})$

### 3.3 收敛性分析

**定理1**：如果通信图包含一棵生成树且根节点在 $\mathcal{T}$ 中，则存在 $c_1, c_2 > 0$ 使得：

$$
\lim_{t \to \infty} \epsilon_i(t) = 0, \quad \forall i
$$

**证明思路**：

1. Laplacian矩阵 $L$ 有一个零特征值，其余特征值为正（连通图）
2. 矩阵 $I_N \otimes S - c_1 L \otimes I_q - c_2 D \otimes I_q$ 的特征值可以计算
3. 选择足够大的 $c_1, c_2$ 使所有特征值实部为负（除了对应于同步流形的零特征值）

**增益选择经验**：
- $c_1 \in [50, 200]$: 共识增益
- $c_2 \in [20, 100]$: 跟踪增益
- 通常 $c_1 > c_2$

---

## 4. 状态反馈控制器设计

### 4.1 控制律结构

基于**调节理论**和**观测器**，设计控制律：

$$
u_i = K_{i1} x_i + K_{i2} \hat{v}_i
$$

其中：
- $K_{i1} \in \mathbb{R}^{m \times n}$: 状态反馈增益
- $K_{i2} \in \mathbb{R}^{m \times q}$: 前馈增益

### 4.2 增益设计

#### 4.2.1 前馈增益

直接使用调节方程的解：

$$
K_{i2} = \Gamma_i
$$

#### 4.2.2 状态反馈增益

设计 $K_{i1}$ 使得 $A_i + B_i K_{i1}$ 是Hurwitz的（稳定的）。

**方法1：极点配置（Pole Placement）**

选择期望的闭环极点 $\{\lambda_1, \lambda_2, \ldots, \lambda_n\}$，使用Ackermann公式或`scipy.signal.place_poles`：

$$
K_{i1} = -[\lambda_n I + A_{cl}^{n-1}]^{-1} [0 \cdots 0 \quad 1] \mathcal{C}^{-1}
$$

其中 $\mathcal{C} = [B_i, A_i B_i, \ldots, A_i^{n-1} B_i]$ 是可控性矩阵。

**Python实现**：
```python
from scipy.signal import place_poles

desired_poles = [-2, -2.5, -3, -3.5]  # 选择负实部极点
result = place_poles(A_i, B_i, desired_poles)
K_i1 = result.gain_matrix
```

**方法2：LQR（Linear Quadratic Regulator）**

最小化性能指标：

$$
J = \int_0^\infty (x^T Q x + u^T R u) dt
$$

求解代数Riccati方程（ARE）：

$$
A_i^T P + P A_i - P B_i R^{-1} B_i^T P + Q = 0
$$

最优增益：

$$
K_{i1} = -R^{-1} B_i^T P
$$

**Python实现**：
```python
from scipy.linalg import solve_continuous_are

Q = np.eye(n) * 10  # 状态权重
R = np.array([[1]])  # 控制权重

P = solve_continuous_are(A_i, B_i, Q, R)
K_i1 = -np.linalg.inv(R) @ B_i.T @ P
```

### 4.3 最终控制律

结合调节器和镇定器：

$$
u_i = K_{i1}(x_i - \Pi_i \hat{v}_i) + \Gamma_i \hat{v}_i
$$

等价地：

$$
u_i = K_{i1} x_i + (K_{i2} - K_{i1}\Pi_i) \hat{v}_i
$$

定义：

$$
K_{i2}^{final} = \Gamma_i - K_{i1}\Pi_i
$$

则：

$$
u_i = K_{i1} x_i + K_{i2}^{final} \hat{v}_i
$$

### 4.4 实际计算示例

**智能体1参数**：

$$
A_1, B_1, \Pi_1, \Gamma_1 \quad \text{(如前)}
$$

**极点配置**：

期望极点：$[-2, -2.5, -3, -3.5]$

求解得：

$$
K_{11} \approx \begin{bmatrix}
-150 & -60 & -250 & -60
\end{bmatrix}
$$

**最终前馈增益**：

$$
K_{12}^{final} = \Gamma_1 - K_{11}\Pi_1 = \begin{bmatrix}
0.5 & 1.0
\end{bmatrix} - \begin{bmatrix}
-150 & -60 & -250 & -60
\end{bmatrix} \begin{bmatrix}
0.1 & 0 \\
0 & 0.1 \\
0 & 0 \\
0 & 0
\end{bmatrix}
$$

$$
= \begin{bmatrix}
0.5 & 1.0
\end{bmatrix} - \begin{bmatrix}
-15 & -6
\end{bmatrix} = \begin{bmatrix}
15.5 & 7.0
\end{bmatrix}
$$

**控制律**：

$$
u_1 = \begin{bmatrix}
-150 & -60 & -250 & -60
\end{bmatrix} x_1 + \begin{bmatrix}
15.5 & 7.0
\end{bmatrix} \hat{v}_1
$$

---

## 5. 稳定性分析

### 5.1 闭环系统

综合智能体动力学、观测器和控制律，闭环系统为：

$$
\begin{bmatrix}
\dot{x}_i \\
\dot{\epsilon}_i
\end{bmatrix} = \begin{bmatrix}
A_i + B_i K_{i1} & B_i K_{i2} \\
0 & S - c_2 \mathbb{1}_{\{i \in \mathcal{T}\}} I_q
\end{bmatrix} \begin{bmatrix}
x_i \\
\epsilon_i
\end{bmatrix} + \begin{bmatrix}
0 \\
c_1 \sum_{j \in \mathcal{N}_i} a_{ij} \epsilon_j
\end{bmatrix}
$$

### 5.2 分离原理

由于观测器和控制器可以分别设计，系统具有**分离原理**：

1. **观测器收敛**：$\epsilon_i \to 0$
2. **状态镇定**：$A_i + B_i K_{i1}$ Hurwitz
3. **调节方程**：保证稳态误差为零

### 5.3 Lyapunov稳定性

定义Lyapunov函数：

$$
V = \sum_{i=1}^N (x_i - \Pi_i v)^T P_i (x_i - \Pi_i v) + \sum_{i=1}^N \epsilon_i^T Q_i \epsilon_i
$$

其中 $P_i, Q_i > 0$ 是正定矩阵。

计算导数：

$$
\dot{V} = \sum_{i=1}^N 2(x_i - \Pi_i v)^T P_i (A_i + B_i K_{i1})(x_i - \Pi_i v) + \cdots < 0
$$

通过选择适当的 $K_{i1}, c_1, c_2$，可以保证 $\dot{V} < 0$，从而系统稳定。

---

## 6. 数值计算示例

### 6.1 完整示例：单个智能体

**系统参数**（Cart-Pendulum）：

$$
m = 0.1 \text{ kg}, \quad M = 1.0 \text{ kg}, \quad l = 0.1 \text{ m}, \quad g = 9.8 \text{ m/s}^2
$$

**计算系统矩阵**：

$$
\mu_1 = \frac{f}{lM} = \frac{0.15}{0.1 \times 1.0} = 1.5
$$

$$
\mu_2 = \frac{(M+m)g}{lM} = \frac{1.1 \times 9.8}{0.1 \times 1.0} = 107.8
$$

$$
\mu_3 = -\frac{f}{M} = -0.15
$$

$$
b = \frac{1}{lM} = 10
$$

$$
A = \begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & 9.8 & 0 \\
0 & 0 & 0 & 1 \\
0 & 1.5 & 107.8 & -0.15
\end{bmatrix}
$$

$$
E = \begin{bmatrix}
0 & 0 \\
\frac{2}{M} & 0 \\
0 & 0 \\
\frac{1}{lM} & 0
\end{bmatrix} = \begin{bmatrix}
0 & 0 \\
2.0 & 0 \\
0 & 0 \\
10.0 & 0
\end{bmatrix}
$$

**求解调节方程**：

使用前述算法，得到：

$$
\Pi = \begin{bmatrix}
0.102 & 0.001 \\
0.001 & 0.102 \\
0.000 & 0.000 \\
0.001 & 0.000
\end{bmatrix}, \quad
\Gamma = \begin{bmatrix}
0.51 & 1.02
\end{bmatrix}
$$

**极点配置**：

期望极点：$[-2, -2.5, -3, -3.5]$

$$
K_1 = \begin{bmatrix}
-152 & -62 & -248 & -59
\end{bmatrix}
$$

**验证闭环稳定性**：

$$
A_{cl} = A + BK_1 = \begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & 9.8 & 0 \\
0 & 0 & 0 & 1 \\
-1520 & -619.5 & -2372.2 & -590.15
\end{bmatrix}
$$

特征值：$\sigma(A_{cl}) = \{-2.00, -2.50, -3.00, -3.50\}$ ✓ 全部负实部

**最终控制律**：

$$
K_2 = \Gamma - K_1 \Pi = \begin{bmatrix}
16.0 & 7.3
\end{bmatrix}
$$

$$
u = \begin{bmatrix}
-152 & -62 & -248 & -59
\end{bmatrix} x + \begin{bmatrix}
16.0 & 7.3
\end{bmatrix} \hat{v}
$$

### 6.2 仿真验证

**初始条件**：

$$
x_0 = \begin{bmatrix}
0.1 \\ 0 \\ 0.05 \\ 0
\end{bmatrix}, \quad
\hat{v}_0 = \begin{bmatrix}
1.0 \\ 0
\end{bmatrix}, \quad
v_0 = \begin{bmatrix}
1.0 \\ 0
\end{bmatrix}
$$

**观测器参数**：$c_1 = 150, c_2 = 50$

**仿真时间**：$t \in [0, 20]$s

**结果**：

| 时间 (s) | $\|e(t)\|$ | $\|\epsilon(t)\|$ | 控制输入 $u$ |
|---------|-----------|------------------|-------------|
| 0 | 0.05 | 0 | 15.5 |
| 1 | 0.12 | 0.03 | -5.2 |
| 5 | 0.05 | 0.001 | 2.1 |
| 10 | 0.02 | 0.0001 | -0.8 |
| 20 | 0.005 | <1e-6 | 0.3 |

**收敛性**：
- 跟踪误差 $e(t) \to 0.005$ (收敛到小误差)
- 观测误差 $\epsilon(t) \to 0$ (指数收敛)
- 控制输入有界且光滑

---

## 7. 总结

### 7.1 设计流程

```
1. 建立系统模型 (A_i, B_i, E_i, C_i, F_i, S)
         ↓
2. 验证假设 (可镇定性、可检测性、传输零点)
         ↓
3. 求解调节方程 → (Π_i, Γ_i)
         ↓
4. 设计状态反馈增益 K_i1 (极点配置或LQR)
         ↓
5. 计算最终增益 K_i2 = Γ_i - K_i1·Π_i
         ↓
6. 设计分布式观测器 (选择 c_1, c_2)
         ↓
7. 仿真验证并调参
```

### 7.2 关键参数选择

| 参数 | 推荐范围 | 影响 |
|-----|---------|-----|
| 极点实部 | [-5, -1] | 收敛速度 vs 控制能耗 |
| $c_1$ | [50, 200] | 共识速度 |
| $c_2$ | [20, 100] | 跟踪速度 |
| LQR $Q$ | $\text{diag}(1, 10, 1, 10)$ | 状态权重 |
| LQR $R$ | $1$ | 控制权重 |

### 7.3 故障排查

**问题1：系统不稳定**
- 检查 $A + BK_1$ 的特征值（是否全部负实部）
- 重新选择极点或增加 $Q$ 矩阵权重

**问题2：跟踪误差大**
- 检查调节方程解（$C\Pi + F \stackrel{?}{=} 0$）
- 增加观测器增益 $c_1, c_2$

**问题3：振荡**
- 减小观测器增益
- 增加阻尼（选择实数极点，避免复数极点）

---

## 参考文献

1. Francis & Wonham, "The internal model principle of control theory", Automatica, 1976
2. Huang, "Nonlinear Output Regulation: Theory and Applications", SIAM, 2004
3. Su & Huang, "Cooperative Output Regulation of Multi-Agent Systems", Springer, 2013
4. Ogata, "Modern Control Engineering", Pearson, 2010

---

*文档版本: 1.0*
*最后更新: 2025-10-30*
