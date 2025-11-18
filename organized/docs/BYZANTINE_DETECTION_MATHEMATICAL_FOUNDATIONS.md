# Byzantine检测方法的数学基础

**文档目的**: 详细讲解本研究中使用的三种Byzantine节点检测方法的数学原理、模型和计算细节

**作者**: liziyu
**日期**: 2025-10-30
**难度**: 研究生水平（包含详细推导）

---

## 目录

1. [系统模型与问题定义](#1-系统模型与问题定义)
2. [方法1: RCP-f距离过滤算法](#2-方法1-rcp-f距离过滤算法)
3. [方法2: ℓ1数据驱动检测](#3-方法2-ℓ1数据驱动检测)
4. [方法3: LSTM机器学习检测](#4-方法3-lstm机器学习检测)
5. [方法对比与互补性分析](#5-方法对比与互补性分析)
6. [实际计算示例](#6-实际计算示例)

---

## 1. 系统模型与问题定义

### 1.1 多智能体系统数学模型

考虑一个由 $N=8$ 个异构智能体组成的系统，每个智能体 $i$ 的动力学方程为：

$$
\begin{cases}
\dot{x}_i(t) = A_i x_i(t) + B_i u_i(t) + E_i v(t) \\
y_i(t) = C_i x_i(t) + F_i v(t)
\end{cases}
$$

**变量定义**：
- $x_i \in \mathbb{R}^{n}$: 智能体 $i$ 的状态向量（$n=4$，包含位置、速度、角度、角速度）
- $u_i \in \mathbb{R}^{m}$: 控制输入（$m=1$）
- $v(t) \in \mathbb{R}^{q}$: 参考信号（$q=2$）
- $y_i \in \mathbb{R}^{p}$: 输出（$p=1$）

**系统矩阵**（以cart-pendulum为例）：

$$
A_i = \begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & g & 0 \\
0 & 0 & 0 & 1 \\
0 & \frac{\mu_{i1}}{l_i M_i} & \frac{(M_i+m_i)g}{l_i M_i} & \frac{-f_i}{M_i}
\end{bmatrix}, \quad
B_i = \begin{bmatrix}
0 \\ 0 \\ 0 \\ \frac{1}{l_i M_i}
\end{bmatrix}
$$

$$
E_i = \begin{bmatrix}
0 & 0 \\
\frac{2}{M_i} & 0 \\
0 & 0 \\
\frac{1}{l_i M_i} & 0
\end{bmatrix}, \quad
C_i = \begin{bmatrix}
1 & 0 & -l_i & 0
\end{bmatrix}, \quad
F_i = \begin{bmatrix}
-1 & 0
\end{bmatrix}
$$

其中：
- $m_i$: 摆的质量
- $M_i$: 小车质量
- $l_i$: 摆长
- $g = 9.8$: 重力加速度
- $f_i$: 摩擦系数

### 1.2 参考信号模型

参考信号服从线性外系统：

$$
\dot{v}(t) = S v(t), \quad v(0) = v_0
$$

其中：

$$
S = \begin{bmatrix}
0 & 1 \\
-1 & 0
\end{bmatrix}
$$

这产生一个圆周运动：

$$
v(t) = \begin{bmatrix}
\cos(t) \\
\sin(t)
\end{bmatrix}
$$

### 1.3 通信拓扑

智能体之间的通信用无向图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 表示：

- $\mathcal{V} = \{1, 2, \ldots, N\}$: 节点集合
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$: 边集合

**邻接矩阵** $\mathcal{A} = [a_{ij}] \in \{0,1\}^{N \times N}$：

$$
a_{ij} = \begin{cases}
1, & \text{if } (i,j) \in \mathcal{E} \\
0, & \text{otherwise}
\end{cases}
$$

**邻居集合**：

$$
\mathcal{N}_i = \{j \in \mathcal{V} : a_{ij} = 1\}
$$

在本系统中，采用部分连接拓扑：

$$
\mathcal{A} = \begin{bmatrix}
0 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\
1 & 0 & 1 & 1 & 0 & 0 & 0 & 0 \\
1 & 1 & 0 & 1 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 & 0 & 1 & 0 & 0 \\
1 & 1 & 1 & 1 & 0 & 0 & 1 & 0 \\
1 & 1 & 1 & 1 & 0 & 0 & 0 & 1 \\
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

### 1.4 Byzantine攻击模型

假设存在 $f \leq f_{\max}$ 个Byzantine节点（本研究中 $f=1$）。

**Byzantine节点行为**：

Byzantine节点 $i \in \mathcal{B}$ （$\mathcal{B}$ 为Byzantine节点集合）可以发送任意错误信息：

$$
\hat{v}_i^{byz}(t) = \hat{v}_i(t) + \Delta_i(t)
$$

其中 $\Delta_i(t)$ 是任意攻击信号，例如：

$$
\Delta_i(t) = \begin{bmatrix}
\alpha \sin(\omega_1 t) + \beta \cos(\omega_2 t) \\
\gamma t
\end{bmatrix}
$$

在实验中使用：$\alpha=50, \beta=15, \gamma=1/15, \omega_1=10, \omega_2=12$

### 1.5 控制目标

**合作输出调节问题**：设计分布式控制律 $u_i(t)$，使得：

1. **跟踪**: $\lim_{t \to \infty} e_i(t) = 0$，其中 $e_i(t) = y_i(t) - y_{ref}(t)$
2. **一致性**: $\lim_{t \to \infty} (\hat{v}_i(t) - v(t)) = 0$，$\forall i \notin \mathcal{B}$

---

## 2. 方法1: RCP-f距离过滤算法

### 2.1 算法原理

**核心思想**：Byzantine节点发送的估计值 $\hat{v}_i$ 会显著偏离正常节点的估计值。通过计算距离并移除最远的 $f$ 个邻居，可以过滤掉Byzantine影响。

### 2.2 数学模型

#### 步骤1: 距离计算

对于节点 $i$，计算其与所有邻居 $j \in \mathcal{N}_i$ 的欧氏距离：

$$
d_{ij}(t) = \|\hat{v}_i(t) - \hat{v}_j(t)\|_2 = \sqrt{\sum_{k=1}^{q} (\hat{v}_i^{(k)}(t) - \hat{v}_j^{(k)}(t))^2}
$$

其中 $\hat{v}_i^{(k)}$ 是 $\hat{v}_i$ 的第 $k$ 个分量。

#### 步骤2: 排序与过滤

将所有邻居按距离排序：

$$
d_{i,\sigma(1)} \leq d_{i,\sigma(2)} \leq \cdots \leq d_{i,\sigma(|\mathcal{N}_i|)}
$$

其中 $\sigma$ 是排序索引。

**过滤规则**：保留距离最近的 $|\mathcal{N}_i| - f$ 个邻居：

$$
\mathcal{N}_i^{filtered} = \{\sigma(1), \sigma(2), \ldots, \sigma(|\mathcal{N}_i| - f)\}
$$

#### 步骤3: 共识更新

使用过滤后的邻居进行共识更新：

$$
\dot{\hat{v}}_i(t) = S\hat{v}_i(t) + c_1 \sum_{j \in \mathcal{N}_i^{filtered}} (\hat{v}_j(t) - \hat{v}_i(t)) + c_2 \mathbb{1}_{\{i \in \mathcal{T}\}} (v(t) - \hat{v}_i(t))
$$

其中：
- $c_1 > 0$: 共识增益（实验中 $c_1 = 150$）
- $c_2 > 0$: 跟踪增益（实验中 $c_2 = 50$）
- $\mathcal{T}$: 目标节点集合（能直接感知 $v(t)$ 的节点）
- $\mathbb{1}_{\{i \in \mathcal{T}\}}$: 指示函数

### 2.3 理论分析

**定理1（收敛性）**: 如果 $f \leq f_{\max}$，且通信图在移除 $f$ 个节点后仍连通，则对所有正常节点 $i \notin \mathcal{B}$：

$$
\lim_{t \to \infty} \|\hat{v}_i(t) - v(t)\| = 0
$$

**证明思路**（简化版）：

1. Byzantine节点发送的 $\hat{v}_j^{byz}$ 与真实值 $v(t)$ 距离最大
2. 正常节点之间的距离 $d_{ij} = \|\hat{v}_i - \hat{v}_j\|$ 有界
3. 过滤后，Byzantine影响被完全移除
4. 剩余的共识系统等价于无Byzantine情况，由标准共识理论保证收敛

### 2.4 计算复杂度

对于节点 $i$：

1. **距离计算**: $O(|\mathcal{N}_i| \cdot q)$
2. **排序**: $O(|\mathcal{N}_i| \log |\mathcal{N}_i|)$
3. **共识更新**: $O(|\mathcal{N}_i| \cdot q)$

**总复杂度**: $O(|\mathcal{N}_i| \log |\mathcal{N}_i|)$ per time step

### 2.5 优势与局限

**优势**：
- ✅ 实时性强（每步 <0.1ms）
- ✅ 理论保证收敛
- ✅ 无需训练或历史数据
- ✅ 对强攻击鲁棒

**局限**：
- ❌ 需要知道 $f$（Byzantine节点数上界）
- ❌ 无法识别具体哪个节点是Byzantine
- ❌ 对隐蔽攻击（小幅度偏差）敏感度较低

---

## 3. 方法2: ℓ1数据驱动检测

### 3.1 方法来源

**参考文献**: Yan Jiaqi et al., "Secure Data Reconstruction: A Direct Data-Driven Approach"

**核心思想**：使用历史干净数据构建Hankel矩阵，通过 ℓ1 优化重构当前观测数据，从重构残差中识别Byzantine节点。

### 3.2 Hankel矩阵构建

#### 3.2.1 轨迹数据

收集系统在 $T$ 个时间步的输入-输出数据：

$$
\mathbf{w} = \begin{bmatrix}
w(0) \\
w(1) \\
\vdots \\
w(T-1)
\end{bmatrix} \in \mathbb{R}^{T \times d}
$$

其中 $w(t) \in \mathbb{R}^d$ 是时刻 $t$ 的系统状态向量（对所有智能体）。

在本系统中，$d = N \times 6 = 48$（8个智能体，每个6维状态）：

$$
w(t) = \begin{bmatrix}
x_1(t) \\
\hat{v}_1(t) \\
x_2(t) \\
\hat{v}_2(t) \\
\vdots \\
x_8(t) \\
\hat{v}_8(t)
\end{bmatrix}_{48 \times 1}
$$

#### 3.2.2 Hankel矩阵定义

给定窗口长度 $L$（实验中 $L=5$），构建Hankel矩阵：

$$
\mathcal{H}_L(w) = \begin{bmatrix}
w(0)^T & w(1)^T & w(2)^T & \cdots & w(T-L)^T \\
w(1)^T & w(2)^T & w(3)^T & \cdots & w(T-L+1)^T \\
w(2)^T & w(3)^T & w(4)^T & \cdots & w(T-L+2)^T \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
w(L-1)^T & w(L)^T & w(L+1)^T & \cdots & w(T-1)^T
\end{bmatrix}_{d \cdot L \times (T-L+1)}
$$

**矩阵维度**：
- 行数：$d \cdot L = 48 \times 5 = 240$
- 列数：$T - L + 1$（例如 $T=4000$ 时为 3996）

#### 3.2.3 Hankel矩阵的性质

**关键性质**：对于线性时不变系统（LTI），Hankel矩阵的秩等于系统阶数：

$$
\text{rank}(\mathcal{H}_L(w)) = n
$$

其中 $n$ 是系统的McMillan度（总状态维度）。

**直觉理解**：
- Hankel矩阵捕捉了系统的动力学特征
- 干净数据的Hankel矩阵是低秩的
- Byzantine攻击破坏了这种低秩结构

### 3.3 ℓ1优化问题

#### 3.3.1 问题定义

给定参考Hankel矩阵 $H_{ref}$（从历史干净数据构建）和当前观测 $w_{obs}$（可能被Byzantine攻击），求解：

$$
\min_{g \in \mathbb{R}^{T-L+1}} \|w_{obs} - H_{ref} g\|_1
$$

等价地：

$$
\min_{g, r} \quad \sum_{i=1}^{d \cdot L} r_i
$$

$$
\text{subject to} \quad \begin{cases}
-r \leq w_{obs} - H_{ref} g \leq r \\
r \geq 0
\end{cases}
$$

其中 $r = [r_1, r_2, \ldots, r_{d \cdot L}]^T$ 是残差向量。

#### 3.3.2 线性规划转换

将上述问题转换为标准线性规划（LP）形式：

**决策变量**：

$$
x = \begin{bmatrix}
g \\
r
\end{bmatrix} \in \mathbb{R}^{(T-L+1) + d \cdot L}
$$

**目标函数**：

$$
\min \quad c^T x = \begin{bmatrix}
0_{(T-L+1) \times 1} \\
1_{d \cdot L \times 1}
\end{bmatrix}^T x = \sum_{i=1}^{d \cdot L} r_i
$$

**约束条件**：

$$
A_{ub} x \leq b_{ub}
$$

其中：

$$
A_{ub} = \begin{bmatrix}
H_{ref} & -I_{d \cdot L} \\
-H_{ref} & -I_{d \cdot L}
\end{bmatrix}_{2(d \cdot L) \times ((T-L+1) + d \cdot L)}, \quad
b_{ub} = \begin{bmatrix}
w_{obs} \\
-w_{obs}
\end{bmatrix}_{2(d \cdot L) \times 1}
$$

#### 3.3.3 求解算法

使用 **单纯形法** 或 **内点法** 求解：

**Scipy实现**：
```python
from scipy.optimize import linprog

result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
g_optimal = result.x[:T-L+1]
```

**时间复杂度**：$O((d \cdot L)^3)$ in worst case，实际中约 1-2ms

### 3.4 Byzantine检测

#### 3.4.1 重构数据

使用最优 $g^*$ 重构干净数据：

$$
w_{clean} = H_{ref} g^*
$$

#### 3.4.2 残差计算

计算每个维度的残差：

$$
\text{residual}_i = |w_{obs}^{(i)} - w_{clean}^{(i)}|, \quad i = 1, 2, \ldots, d \cdot L
$$

#### 3.4.3 Agent级别残差

将残差映射回各个智能体（每个智能体占6维）：

$$
R_{\text{agent}_j} = \frac{1}{6L} \sum_{k \in \text{indices of agent } j} \text{residual}_k
$$

#### 3.4.4 检测规则

设定阈值 $\tau$（实验中 $\tau = 0.1$）：

$$
\mathcal{B}_{detected} = \{j : R_{\text{agent}_j} > \tau\}
$$

### 3.5 理论分析

**定理2（重构保证）**: 如果Byzantine攻击是稀疏的（至多 $s$ 个维度被攻击），且 $H_{ref}$ 满足Restricted Isometry Property (RIP)，则ℓ1优化能以高概率恢复干净数据。

**RIP条件**：存在常数 $\delta_s \in (0,1)$，使得对所有 $s$-稀疏向量 $x$：

$$
(1 - \delta_s)\|x\|_2^2 \leq \|H_{ref} x\|_2^2 \leq (1 + \delta_s)\|x\|_2^2
$$

### 3.6 优势与局限

**优势**：
- ✅ 无需系统模型
- ✅ 理论保证（凸优化）
- ✅ 可处理批量数据
- ✅ 对稀疏攻击有效

**局限**：
- ❌ 需要历史干净数据
- ❌ 计算开销大（1-2ms per detection）
- ❌ 实时性差（无法用于控制环）
- ❌ 对非稀疏攻击效果差

---

## 4. 方法3: LSTM机器学习检测

### 4.1 问题建模

**目标**：学习一个函数 $f: \mathcal{X} \to \{0, 1\}$，将时间序列特征映射到二分类标签：

$$
\hat{y} = f(X; \theta), \quad \hat{y} \in \{0, 1\}
$$

其中：
- $X = [x_1, x_2, \ldots, x_T] \in \mathbb{R}^{T \times d}$: 滑动窗口特征序列
- $\theta$: LSTM模型参数
- $\hat{y} = 1$: Byzantine行为
- $\hat{y} = 0$: 正常行为

### 4.2 特征工程

#### 4.2.1 基础特征（7维）

对于每个智能体 $i$，在时刻 $t$ 提取：

$$
x_t^{(i)} = \begin{bmatrix}
\text{estimation\_error}_i(t) \\
\text{position\_error}_i(t) \\
\theta_i(t) \\
\dot{\theta}_i(t) \\
u_i(t) \\
\hat{v}_{i,1}(t) \\
\hat{v}_{i,2}(t)
\end{bmatrix}_{7 \times 1}
$$

其中：

$$
\text{estimation\_error}_i(t) = \|\hat{v}_i(t) - v(t)\|_2 = \sqrt{(\hat{v}_{i,1}(t) - \cos t)^2 + (\hat{v}_{i,2}(t) - \sin t)^2}
$$

$$
\text{position\_error}_i(t) = |y_i(t) - y_{ref}(t)|
$$

#### 4.2.2 Correntropy增强特征（3维）

**Maximum Correntropy Criterion (MCC)** 定义：

$$
V(X, Y) = \mathbb{E}[\kappa_\sigma(X - Y)]
$$

其中 **高斯核**：

$$
\kappa_\sigma(e) = G_\sigma(e) = \exp\left(-\frac{\|e\|^2}{2\sigma^2}\right)
$$

**物理意义**：
- $\kappa_\sigma(e) \approx 1$: $X$ 和 $Y$ 相似
- $\kappa_\sigma(e) \approx 0$: $X$ 和 $Y$ 差异大（outlier）

**Correntropy展开式**（Taylor展开）：

$$
G_\sigma(e) = \sum_{n=0}^{\infty} \frac{(-1)^n}{2^n \cdot n! \cdot \sigma^{2n}} \mathbb{E}[e^{2n}]
$$

这表明Correntropy **包含所有偶数阶矩**，比欧氏距离（仅二阶矩）更全面。

**计算Correntropy特征**：

对于智能体 $i$，计算其与所有其他智能体 $j \neq i$ 的相似度：

$$
\text{corr}_{ij}(t) = \exp\left(-\frac{\|\hat{v}_i(t) - \hat{v}_j(t)\|^2}{2\sigma^2}\right)
$$

**自适应 $\sigma$ 选择**：

$$
\sigma(t) = \max\left(\text{median}\{d_{ij}(t) : j \neq i\}, 0.1\right)
$$

其中 $d_{ij}(t) = \|\hat{v}_i(t) - \bar{v}(t)\|$，$\bar{v}(t) = \frac{1}{N}\sum_{j=1}^{N} \hat{v}_j(t)$

**三个Correntropy特征**：

$$
\text{avg\_correntropy}_i(t) = \frac{1}{N-1} \sum_{j \neq i} \text{corr}_{ij}(t)
$$

$$
\text{min\_correntropy}_i(t) = \min_{j \neq i} \text{corr}_{ij}(t)
$$

$$
\text{std\_correntropy}_i(t) = \sqrt{\frac{1}{N-1} \sum_{j \neq i} (\text{corr}_{ij}(t) - \text{avg\_correntropy}_i(t))^2}
$$

**直觉理解**：
- Normal节点：与大家相似 → `avg_corr` 高（0.7-0.9）
- Byzantine节点：与大家不同 → `avg_corr` 低（0.1-0.3）
- `min_corr` 捕捉最不相似的邻居
- `std_corr` 反映相似度的波动

#### 4.2.3 最终特征向量（10维）

$$
x_t^{(i)} = \begin{bmatrix}
\text{基础7维特征} \\
\text{avg\_correntropy}_i(t) \\
\text{min\_correntropy}_i(t) \\
\text{std\_correntropy}_i(t)
\end{bmatrix}_{10 \times 1}
$$

### 4.3 LSTM模型架构

#### 4.3.1 LSTM单元数学定义

标准LSTM单元在时刻 $t$ 的计算：

**输入门**：

$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
$$

**遗忘门**：

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
$$

**输出门**：

$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
$$

**候选记忆单元**：

$$
\tilde{C}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
$$

**记忆单元更新**：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

**隐藏状态**：

$$
h_t = o_t \odot \tanh(C_t)
$$

其中：
- $\sigma(\cdot)$: sigmoid函数，$\sigma(z) = \frac{1}{1 + e^{-z}}$
- $\tanh(\cdot)$: 双曲正切函数
- $\odot$: Hadamard积（逐元素乘法）
- $W_*, U_*, b_*$: 可学习参数

#### 4.3.2 完整网络架构

$$
\begin{align}
&\text{Input:} \quad X = [x_1, x_2, \ldots, x_T] \in \mathbb{R}^{T \times 10} \\
&\downarrow \\
&\text{LSTM Layer:} \quad h_t = \text{LSTM}(x_t, h_{t-1}, C_{t-1}; \theta_{LSTM}) \in \mathbb{R}^{32} \\
&\downarrow \\
&\text{Take last hidden:} \quad h_T \in \mathbb{R}^{32} \\
&\downarrow \\
&\text{FC1:} \quad z_1 = W_1 h_T + b_1 \in \mathbb{R}^{16} \\
&\downarrow \\
&\text{ReLU:} \quad a_1 = \max(0, z_1) \\
&\downarrow \\
&\text{FC2:} \quad z_2 = W_2 a_1 + b_2 \in \mathbb{R}^{2} \\
&\downarrow \\
&\text{Softmax:} \quad \hat{p} = \text{softmax}(z_2) = \begin{bmatrix} p_0 \\ p_1 \end{bmatrix}
\end{align}
$$

其中 softmax 定义为：

$$
\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{2} e^{z_j}}, \quad k \in \{0, 1\}
$$

**预测**：

$$
\hat{y} = \arg\max_{k} p_k = \begin{cases}
0, & \text{if } p_0 > p_1 \quad (\text{Normal}) \\
1, & \text{if } p_1 > p_0 \quad (\text{Byzantine})
\end{cases}
$$

### 4.4 训练过程

#### 4.4.1 损失函数

**交叉熵损失**：

$$
\mathcal{L}(\theta) = -\frac{1}{M} \sum_{m=1}^{M} \left[ y^{(m)} \log(p_1^{(m)}) + (1 - y^{(m)}) \log(p_0^{(m)}) \right]
$$

其中：
- $M$: batch size（实验中32）
- $y^{(m)} \in \{0, 1\}$: 真实标签
- $p_0^{(m)}, p_1^{(m)}$: 预测概率

#### 4.4.2 优化算法

**Adam优化器**（Adaptive Moment Estimation）：

**一阶矩估计**：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta \mathcal{L}(\theta_t)
$$

**二阶矩估计**：

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta \mathcal{L}(\theta_t))^2
$$

**偏差修正**：

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**参数更新**：

$$
\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

其中：
- $\alpha = 0.001$: 学习率
- $\beta_1 = 0.9, \beta_2 = 0.999$: 矩估计衰减率
- $\epsilon = 10^{-8}$: 数值稳定性项

#### 4.4.3 训练配置

- **Epochs**: 20
- **Batch size**: 32
- **窗口大小**: $T = 50$ 步（1秒 @ 50Hz采样）
- **Stride**: 50步（无重叠）
- **训练集**: 70%
- **验证集**: 15%
- **测试集**: 15%

### 4.5 在线检测

#### 4.5.1 滑动窗口

维护一个FIFO缓冲区 $\mathcal{B}_i$ for each agent $i$：

$$
\mathcal{B}_i(t) = [x_{t-T+1}^{(i)}, x_{t-T+2}^{(i)}, \ldots, x_{t}^{(i)}]
$$

当 $|\mathcal{B}_i| = T$，执行检测：

$$
\hat{y}_i(t) = f(\mathcal{B}_i(t); \theta^*)
$$

#### 4.5.2 置信度输出

$$
\text{confidence}_i(t) = p_1^{(i)}(t) = P(y_i = 1 | \mathcal{B}_i(t))
$$

**检测规则**：

$$
i \in \mathcal{B}_{detected} \quad \Leftrightarrow \quad \text{confidence}_i(t) > 0.5
$$

### 4.6 优势与局限

**优势**：
- ✅ 学习复杂的时序模式
- ✅ 可识别具体Byzantine节点
- ✅ 适应多种攻击类型
- ✅ 支持在线实时检测（<1ms）

**局限**：
- ❌ 需要大量训练数据
- ❌ 需要标注数据
- ❌ 训练时间长（~30秒）
- ❌ 可解释性差（黑盒模型）

---

## 5. 方法对比与互补性分析

### 5.1 定量对比表

| 特性 | RCP-f | ℓ1优化 | LSTM |
|-----|-------|--------|------|
| **计算复杂度** | $O(N \log N)$ | $O((dL)^3)$ | $O(dH^2)$ |
| **实时性（ms）** | <0.1 | 1-2 | <1 |
| **是否需要训练** | ❌ | ❌ | ✅ |
| **是否需要历史数据** | ❌ | ✅ | ✅ |
| **理论保证** | ✅ | ✅ | ❌ |
| **可识别具体节点** | ❌ | ✅ | ✅ |
| **对隐蔽攻击** | 中 | 高 | 高 |
| **对强攻击** | 高 | 中 | 高 |

其中 $H$ 是LSTM隐藏层维度（32）。

### 5.2 互补性分析

#### 5.2.1 工作阶段互补

```
时间轴: |-----离线阶段-----|-----在线运行-----|

方法1 (ℓ1):       [批量验证]         [不适用]
方法2 (LSTM):     [训练]        [实时检测，每50步]
方法3 (RCP-f):    [不需要]      [实时过滤，每步]
```

#### 5.2.2 检测能力互补

**攻击类型覆盖**：

| 攻击类型 | RCP-f | ℓ1 | LSTM | 组合 |
|---------|-------|-----|------|------|
| 强幅度攻击 | 优秀 | 良好 | 优秀 | 优秀 |
| 隐蔽攻击 | 一般 | 优秀 | 优秀 | 优秀 |
| 稀疏攻击 | 良好 | 优秀 | 良好 | 优秀 |
| 持续攻击 | 优秀 | 一般 | 优秀 | 优秀 |

#### 5.2.3 数学角度互补

**RCP-f（几何方法）**：

$$
\text{Decision} = f_{\text{geom}}(\{d_{ij}\}_{j \in \mathcal{N}_i})
$$

基于距离度量，几何空间判断。

**ℓ1（优化方法）**：

$$
\text{Decision} = f_{\text{opt}}(\text{argmin}_{g} \|w - Hg\|_1)
$$

基于凸优化，代数空间判断。

**LSTM（统计方法）**：

$$
\text{Decision} = f_{\text{stat}}(\{x_t\}_{t=1}^T; \theta^*)
$$

基于概率统计，学习空间判断。

### 5.3 组合策略

#### 5.3.1 两阶段检测

**阶段1（离线）**：使用 ℓ1 验证历史数据质量

$$
\mathcal{B}_{\ell_1} = \{i : R_i > \tau_1\}
$$

**阶段2（在线）**：LSTM检测 + RCP-f过滤

$$
\mathcal{B}_{LSTM}(t) = \{i : p_1^{(i)}(t) > 0.5\}
$$

$$
\mathcal{B}_{total}(t) = \mathcal{B}_{\ell_1} \cup \mathcal{B}_{LSTM}(t)
$$

在共识更新中排除 $\mathcal{B}_{total}$：

$$
\mathcal{N}_i^{final} = \mathcal{N}_i^{filtered} \setminus \mathcal{B}_{total}
$$

#### 5.3.2 置信度融合

**加权融合**：

$$
\text{score}_i(t) = w_1 \cdot \mathbb{1}_{i \in \mathcal{B}_{\ell_1}} + w_2 \cdot p_1^{(i)}(t) + w_3 \cdot \left(1 - \frac{d_i^{sorted}(t)}{d_{max}(t)}\right)
$$

其中：
- $w_1, w_2, w_3$: 权重（例如 $0.3, 0.4, 0.3$）
- $d_i^{sorted}$: 节点 $i$ 在RCP-f排序中的位置
- $d_{max}$: 最大距离

**最终决策**：

$$
i \in \mathcal{B}_{final} \quad \Leftrightarrow \quad \text{score}_i(t) > \tau_{fusion}
$$

---

## 6. 实际计算示例

### 6.1 RCP-f计算示例

**场景**：8个智能体，节点0是Byzantine，$f=1$

**时刻** $t=5.0$s，节点4的估计值：

$$
\hat{v}_4 = \begin{bmatrix} 0.28 \\ -0.96 \end{bmatrix}
$$

**邻居估计值**：

$$
\begin{align}
\hat{v}_0 &= \begin{bmatrix} 35.2 \\ 2.5 \end{bmatrix} \quad (\text{Byzantine}) \\
\hat{v}_1 &= \begin{bmatrix} 0.30 \\ -0.95 \end{bmatrix} \\
\hat{v}_2 &= \begin{bmatrix} 0.27 \\ -0.97 \end{bmatrix} \\
\hat{v}_3 &= \begin{bmatrix} 0.29 \\ -0.96 \end{bmatrix} \\
\hat{v}_5 &= \begin{bmatrix} 0.31 \\ -0.95 \end{bmatrix}
\end{align}
$$

**步骤1：计算距离**

$$
\begin{align}
d_{4,0} &= \sqrt{(35.2-0.28)^2 + (2.5-(-0.96))^2} = \sqrt{1223.9} = 34.98 \\
d_{4,1} &= \sqrt{(0.30-0.28)^2 + (-0.95-(-0.96))^2} = 0.022 \\
d_{4,2} &= \sqrt{(0.27-0.28)^2 + (-0.97-(-0.96))^2} = 0.014 \\
d_{4,3} &= \sqrt{(0.29-0.28)^2 + (-0.96-(-0.96))^2} = 0.010 \\
d_{4,5} &= \sqrt{(0.31-0.28)^2 + (-0.95-(-0.96))^2} = 0.032
\end{align}
$$

**步骤2：排序**

$$
d_{4,3} < d_{4,2} < d_{4,1} < d_{4,5} < d_{4,0}
$$

排序索引：$\sigma = [3, 2, 1, 5, 0]$

**步骤3：过滤**（保留 $|\mathcal{N}_4| - f = 5 - 1 = 4$ 个）

$$
\mathcal{N}_4^{filtered} = \{3, 2, 1, 5\}
$$

节点0（Byzantine）被成功过滤！

**步骤4：共识更新**

$$
\begin{align}
\text{filtered\_mean} &= \frac{1}{4}(\hat{v}_1 + \hat{v}_2 + \hat{v}_3 + \hat{v}_5) \\
&= \frac{1}{4}\left(\begin{bmatrix}0.30\\-0.95\end{bmatrix} + \begin{bmatrix}0.27\\-0.97\end{bmatrix} + \begin{bmatrix}0.29\\-0.96\end{bmatrix} + \begin{bmatrix}0.31\\-0.95\end{bmatrix}\right) \\
&= \begin{bmatrix}0.293\\-0.958\end{bmatrix}
\end{align}
$$

$$
\begin{align}
\dot{\hat{v}}_4 &= S\hat{v}_4 + 150(\text{filtered\_mean} - \hat{v}_4) + 50(v(5) - \hat{v}_4) \\
&= \begin{bmatrix}0&1\\-1&0\end{bmatrix}\begin{bmatrix}0.28\\-0.96\end{bmatrix} + 150\left(\begin{bmatrix}0.293\\-0.958\end{bmatrix} - \begin{bmatrix}0.28\\-0.96\end{bmatrix}\right) \\
&\quad + 50\left(\begin{bmatrix}0.284\\-0.959\end{bmatrix} - \begin{bmatrix}0.28\\-0.96\end{bmatrix}\right)
\end{align}
$$

### 6.2 ℓ1优化计算示例

**简化场景**：3个智能体，每个2维状态，$L=2$

**干净参考数据**（$T=5$ 步）：

$$
w_{ref} = \begin{bmatrix}
0.0 & 1.0 & 0.0 & 1.0 & 0.0 & 1.0 \\
1.0 & 0.0 & 1.0 & 0.0 & 1.0 & 0.0 \\
0.0 & -1.0 & 0.0 & -1.0 & 0.0 & -1.0 \\
-1.0 & 0.0 & -1.0 & 0.0 & -1.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 1.0 & 0.0 & 1.0
\end{bmatrix}_{5 \times 6}
$$

每行是一个时间步，每2列是一个智能体的状态。

**Hankel矩阵**（$d=6, L=2$）：

$$
H_{ref} = \begin{bmatrix}
0.0 & 1.0 & 0.0 & 1.0 & 0.0 & 1.0 & | & 1.0 & 0.0 & \cdots \\
1.0 & 0.0 & 1.0 & 0.0 & 1.0 & 0.0 & | & 0.0 & -1.0 & \cdots \\
\hline
1.0 & 0.0 & 1.0 & 0.0 & 1.0 & 0.0 & | & 0.0 & -1.0 & \cdots \\
0.0 & -1.0 & 0.0 & -1.0 & 0.0 & -1.0 & | & -1.0 & 0.0 & \cdots
\end{bmatrix}_{12 \times 4}
$$

（为简化显示省略部分列）

**当前观测**（智能体1被攻击）：

$$
w_{obs} = [0.0, 1.0, \mathbf{10.0}, \mathbf{5.0}, 0.0, 1.0, 1.0, 0.0, \mathbf{12.0}, \mathbf{6.0}, 0.0, -1.0]^T_{12 \times 1}
$$

粗体部分是被攻击的维度。

**ℓ1优化**：

$$
\min_{g \in \mathbb{R}^4} \|w_{obs} - H_{ref} g\|_1
$$

求解后得到：

$$
g^* \approx [0.25, 0.25, 0.25, 0.25]^T
$$

$$
w_{clean} = H_{ref} g^* \approx [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0]^T
$$

**残差**：

$$
r = |w_{obs} - w_{clean}| = [0, 0, \mathbf{10.0}, \mathbf{4.0}, 0, 0, 0, 0, \mathbf{11.0}, \mathbf{6.0}, 0, 0]^T
$$

**智能体级残差**：

$$
\begin{align}
R_0 &= \frac{1}{4}(0 + 0 + 0 + 0) = 0.0 \\
R_1 &= \frac{1}{4}(10.0 + 4.0 + 11.0 + 6.0) = 7.75 \quad (\text{检测到!}) \\
R_2 &= \frac{1}{4}(0 + 0 + 0 + 0) = 0.0
\end{align}
$$

设阈值 $\tau = 0.1$，则 $\mathcal{B}_{detected} = \{1\}$ ✓

### 6.3 LSTM计算示例

**输入序列**（$T=50, d=10$）：

$$
X = \begin{bmatrix}
x_1^T \\
x_2^T \\
\vdots \\
x_{50}^T
\end{bmatrix}_{50 \times 10}
$$

其中 $x_t = [0.05, 0.02, 0.1, 0.01, 2.3, 0.28, -0.96, 0.85, 0.72, 0.08]^T$

**前向传播**（简化，仅展示关键步骤）：

**时刻** $t=1$：

$$
\begin{align}
i_1 &= \sigma(W_i x_1 + U_i h_0 + b_i) \approx [0.5, 0.6, \ldots, 0.4]^T_{32 \times 1} \\
f_1 &= \sigma(W_f x_1 + U_f h_0 + b_f) \approx [0.3, 0.2, \ldots, 0.5]^T_{32 \times 1} \\
o_1 &= \sigma(W_o x_1 + U_o h_0 + b_o) \approx [0.6, 0.7, \ldots, 0.5]^T_{32 \times 1} \\
\tilde{C}_1 &= \tanh(W_c x_1 + U_c h_0 + b_c) \approx [0.1, 0.2, \ldots, 0.05]^T_{32 \times 1} \\
C_1 &= f_1 \odot C_0 + i_1 \odot \tilde{C}_1 \approx [0.05, 0.12, \ldots, 0.02]^T \\
h_1 &= o_1 \odot \tanh(C_1) \approx [0.03, 0.08, \ldots, 0.01]^T_{32 \times 1}
\end{align}
$$

**时刻** $t=50$（最后一步）：

$$
h_{50} \approx [0.42, -0.15, 0.68, \ldots, 0.23]^T_{32 \times 1}
$$

**全连接层**：

$$
\begin{align}
z_1 &= W_1 h_{50} + b_1 \in \mathbb{R}^{16} \\
&\approx [1.2, -0.5, 2.1, 0.3, \ldots, 0.8]^T \\
a_1 &= \text{ReLU}(z_1) = [1.2, 0, 2.1, 0.3, \ldots, 0.8]^T \\
z_2 &= W_2 a_1 + b_2 \in \mathbb{R}^2 \\
&\approx \begin{bmatrix} -1.5 \\ 2.3 \end{bmatrix}
\end{align}
$$

**Softmax**：

$$
\begin{align}
p_0 &= \frac{e^{-1.5}}{e^{-1.5} + e^{2.3}} = \frac{0.223}{0.223 + 9.974} = 0.022 \\
p_1 &= \frac{e^{2.3}}{e^{-1.5} + e^{2.3}} = \frac{9.974}{10.197} = 0.978
\end{align}
$$

$$
\hat{p} = \begin{bmatrix} 0.022 \\ 0.978 \end{bmatrix}
$$

**预测**：$\hat{y} = 1$ (Byzantine)，置信度 = 97.8% ✓

---

## 7. 总结

### 7.1 方法特点总结

| 方法 | 核心思想 | 数学基础 | 适用场景 |
|-----|---------|---------|---------|
| **RCP-f** | 距离最远的是Byzantine | 几何度量 | 实时控制 |
| **ℓ1** | 低秩+稀疏性 | 凸优化 | 离线验证 |
| **LSTM** | 学习行为模式 | 深度学习 | 在线检测 |

### 7.2 数学工具箱

本研究使用的数学工具：

1. **线性代数**: 矩阵运算、Hankel矩阵、秩理论
2. **优化理论**: 凸优化、线性规划、ℓ1范数最小化
3. **概率统计**: 高斯核、Correntropy、概率分类
4. **机器学习**: LSTM、反向传播、Adam优化
5. **控制理论**: LTI系统、共识算法、Lyapunov稳定性

### 7.3 实际应用建议

**选择指南**：

- **仅需实时防御** → RCP-f（最简单最有效）
- **需要识别节点** → LSTM或ℓ1
- **追求最优性能** → 三者结合
- **计算资源有限** → RCP-f
- **数据充足** → LSTM + RCP-f
- **理论保证重要** → ℓ1 + RCP-f

---

## 参考文献

1. **Yan Jiaqi et al.**, "Secure Data Reconstruction: A Direct Data-Driven Approach", IEEE Transactions, 2023
2. **Luan et al.**, "Maximum Correntropy Criterion-Based Federated Learning", 2025
3. **Hochreiter & Schmidhuber**, "Long Short-Term Memory", Neural Computation, 1997
4. **Boyd & Vandenberghe**, "Convex Optimization", Cambridge University Press, 2004
5. **Goodfellow et al.**, "Deep Learning", MIT Press, 2016

---

*文档版本: 1.0*
*最后更新: 2025-10-30*
*如有疑问，请参考代码实现或联系作者*
