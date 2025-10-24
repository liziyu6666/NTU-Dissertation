import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 读取数据
data = pd.read_csv("C:/Users/ASUS/Desktop/dissertation/code/error_data.csv")
time = data.iloc[:, 0]  # 时间列
error_data = data.iloc[:, 1:]  # 误差数据（去掉时间列）

# 归一化数据
scaler = StandardScaler()
error_data_scaled = scaler.fit_transform(error_data)

# 转换为 PyTorch 张量
seq_length = 10  # LSTM 预测使用的历史时间步
X, y = [], []
for i in range(len(error_data_scaled) - seq_length):
    X.append(error_data_scaled[i:i+seq_length])  # 过去 10 个时间步的数据
    y.append(error_data_scaled[i+seq_length])  # 预测下一个时间步

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

# 创建数据加载器
batch_size = 32
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义 LSTM 模型
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)  # 预测每个 agent 的误差

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的预测结果
        return out

# 初始化模型
input_size = error_data.shape[1]  # 误差数据的特征数
hidden_size = 64
num_layers = 2
model = LSTMPredictor(input_size, hidden_size, num_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练 LSTM
num_epochs = 50
for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 预测下一个时间步
with torch.no_grad():
    predictions = model(X).numpy()
    errors = np.mean((predictions - y.numpy())**2, axis=1)  # 计算 MSE 误差

# 设定异常阈值
threshold = np.percentile(errors, 95)  # 设定异常阈值为 95% 分位数
anomalies = errors > threshold  # 找出异常数据点

# 统计异常情况
anomaly_results = {}
for i, col in enumerate(error_data.columns):
    if anomalies[i]:
        anomaly_results[col] = "Anomaly detected"
    else:
        anomaly_results[col] = "Normal"

print(anomaly_results)
