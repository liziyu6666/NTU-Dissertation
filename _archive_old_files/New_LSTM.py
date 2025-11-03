import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim

# =======================
# 1. 数据生成
# =======================
np.random.seed(42)

# 生成时间序列数据 (sin 波 + 噪声)
time = np.linspace(0, 10, 100)
data = np.sin(time) + np.random.normal(0, 0.1, len(time))

# 20% 的数据缺失
missing_indices = np.random.choice(len(time), size=int(0.2 * len(time)), replace=False)
data_missing = data.copy()
data_missing[missing_indices] = np.nan

# 20% 的数据点受到拜占庭攻击 (恶意篡改)
attack_indices = np.random.choice(len(time), size=int(0.2 * len(time)), replace=False)
data_attack = data.copy()
data_attack[attack_indices] += np.random.normal(2, 0.5, len(attack_indices))  # 增加偏差

# 绘制数据
plt.figure(figsize=(10, 5))
plt.plot(time, data, label="Original Data", linestyle="--")
plt.scatter(time[missing_indices], data_missing[missing_indices], color="red", label="Missing Data")
plt.scatter(time[attack_indices], data_attack[attack_indices], color="purple", label="Byzantine Attack")
plt.legend()
plt.title("Original, Missing, and Byzantine Attack Data")
plt.show()

# =======================
# 2. Algorithm 1: 线性插值恢复缺失值
# =======================
known_indices = np.setdiff1d(np.arange(len(time)), missing_indices)
interpolator = interp1d(known_indices, data[known_indices], kind="linear", fill_value="extrapolate")
data_interpolated = interpolator(np.arange(len(time)))

# =======================
# 3. Algorithm 2: 拜占庭攻击检测
# =======================
threshold = 3  # 设定 Z 分数阈值
mean_val = np.mean(data)
std_val = np.std(data)
z_scores = (data_attack - mean_val) / std_val

# 找出异常值
outliers = np.where(np.abs(z_scores) > threshold)[0]
print("Detected Outliers:", outliers)

# 去除异常值（用均值替换）
data_cleaned = data_attack.copy()
data_cleaned[outliers] = np.mean(data_cleaned)

# =======================
# 4. 使用 LSTM 进行恢复
# =======================

# 数据格式转换 (LSTM 需要时间窗口)
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# 用去除异常后的数据训练 LSTM
X_train, y_train = create_sequences(data_cleaned)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)

# 定义 LSTM 网络
class LSTMReconstructor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMReconstructor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# 训练模型
model = LSTMReconstructor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 预测
predicted = model(X_train).detach().numpy()

# =======================
# 5. 可视化结果
# =======================
plt.figure(figsize=(10, 5))
plt.plot(time, data, label="Original Data", linestyle="--")
plt.plot(time[:len(predicted)], predicted, label="LSTM Reconstructed", linewidth=2)
plt.legend()
plt.title("LSTM Reconstruction of Missing & Byzantine Data")
plt.show()
