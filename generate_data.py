import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# 读取数据
data = pd.read_csv("C:/Users/ASUS/Desktop/dissertation/code/error_data.csv")
time = data.iloc[:, 0]  # 时间列
error_data = data.iloc[:, 1:]  # 误差数据（去掉时间列）

# 数据归一化
scaler = MinMaxScaler()
error_data_scaled = scaler.fit_transform(error_data)

# 逐个 Agent 训练 Isolation Forest 并检测异常
anomaly_results = {}
for i, column in enumerate(error_data.columns):
    model = IsolationForest(contamination=0.1, random_state=42)  # 设定污染率
    model.fit(error_data_scaled[:, i].reshape(-1, 1))
    predictions = model.predict(error_data_scaled[:, i].reshape(-1, 1))
    
    # 判断该 Agent 是否存在异常
    if -1 in predictions:
        anomaly_results[column] = "Anomaly detected"
    else:
        anomaly_results[column] = "Normal"

# 可视化异常检测
plt.figure(figsize=(10, 5))
for i, column in enumerate(error_data.columns):
    plt.plot(time, error_data_scaled[:, i], label=f"{column}")
plt.xlabel("Time")
plt.ylabel("Normalized Error")
plt.legend()
plt.title("Error Trends of Agents")
plt.show()

# 输出检测结果
print(anomaly_results)
