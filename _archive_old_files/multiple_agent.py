import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# 读取数据
data = pd.read_csv("C:/Users/ASUS/Desktop/dissertation/code/error_data.csv")
error_data = data.iloc[:, 1:]  # 误差数据（去掉时间列）

# 标准化数据
scaler = StandardScaler()
error_data_scaled = scaler.fit_transform(error_data)

# 计算每个 Agent 误差的 z-score（标准化后计算）
z_scores = zscore(error_data_scaled, axis=0)  # 计算误差在每个 Agent 中的标准化偏差
threshold = 3  # 设定异常阈值（绝对值超过 3）

# 记录异常 Agent
anomaly_results = {}
for i, column in enumerate(error_data.columns):
    if np.any(np.abs(z_scores[:, i]) > threshold):
        anomaly_results[column] = "Anomaly detected"
    else:
        anomaly_results[column] = "Normal"

# 计算 Agent 之间的相关性
correlation_matrix = np.corrcoef(error_data_scaled.T)  # 计算各 Agent 之间的误差相关性

# 设定相关性阈值，找出异常的 Agent
correlation_threshold = 0.2  # 低相关性代表异常行为
low_correlation_agents = []
for i in range(len(correlation_matrix)):
    if np.mean(correlation_matrix[i]) < correlation_threshold:
        low_correlation_agents.append(error_data.columns[i])

# 更新异常检测结果
for agent in low_correlation_agents:
    anomaly_results[agent] = "Low correlation anomaly"

# 输出异常检测结果
print(anomaly_results)
