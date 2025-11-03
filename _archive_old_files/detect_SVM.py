import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from code import sol_test  # 只使用测试数据
from LSTM import LSTMPredictor

# ========== 加载训练好的 LSTM 模型 ==========
model = LSTMPredictor(input_dim=6)
model.load_state_dict(torch.load("saved_model/lstm_model.pth", weights_only=True))
model.eval()
print("✅ Model loaded successfully!")

# ========== 处理测试数据 ==========
seq_length = 10
X_test = []
y_test = []
num_nodes = sol_test.shape[0]  # 节点数量

for i in range(sol_test.shape[1] - seq_length - 1):
    X_test.append(sol_test[:, i:i+seq_length].T)
    y_test.append(sol_test[:, i+seq_length+1])

X_test = torch.tensor(np.array(X_test), dtype=torch.float32)  # (batch_size, seq_length, num_nodes)
y_test = np.array(y_test)  # (batch_size, num_nodes)
print("✅ Data prepared:", X_test.shape, y_test.shape)

# ========== 进行预测 ==========
with torch.no_grad():
    y_pred = model(X_test).numpy()
print("✅ Prediction completed!")

# ========== 计算各节点的误差 ==========
errors = np.abs(y_test - y_pred)  # 逐个节点的误差
mean_errors = np.mean(errors, axis=0)  # 对所有时间步取均值
var_errors = np.var(errors, axis=0)  # 计算方差

# ========== 识别拜占庭节点 ==========
threshold = np.median(var_errors) * 2  # 经验阈值：误差方差大于中位数的两倍
byzantine_nodes = np.where(var_errors > threshold)[0]

print(f"✅ Detected Byzantine Nodes: {byzantine_nodes}")

# ========== 可视化误差演变 ==========
plt.figure(figsize=(12, 5))
for node in range(num_nodes):
    plt.plot(errors[:, node], label=f"Node {node}" if node in byzantine_nodes else "", alpha=0.6)
plt.xlabel("Time Step")
plt.ylabel("Prediction Error")
plt.title("Error Evolution of Each Node")
plt.legend()
plt.grid()
plt.show()

# ========== 可视化误差分布 ==========
plt.figure(figsize=(10, 5))
plt.boxplot(errors.T, vert=True)
plt.xlabel("Node Index")
plt.ylabel("Prediction Error")
plt.title("Error Distribution Across Nodes")
plt.grid()
plt.show()
