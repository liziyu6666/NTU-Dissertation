import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv("C:/Users/ASUS/Desktop/dissertation/code/error_data.csv")
time = data.iloc[:, 0]  # 时间列
error_data = data.iloc[:, 1:]  # 误差数据（去掉时间列）

# 去掉前几秒的数据（默认去掉前 10%）
trim_ratio = 0.1  # 设定去掉的比例
trim_index = int(len(time) * trim_ratio)
time = time.iloc[trim_index:].reset_index(drop=True)
error_data = error_data.iloc[trim_index:].reset_index(drop=True)

# 重新计算标准化参数
tmp_scaler = StandardScaler()
tmp_scaler.fit(error_data)  # 仅基于去除后的数据计算均值和标准差
error_data_scaled = tmp_scaler.transform(error_data)

# 逐个 Agent 训练 One-Class SVM 并检测异常
anomaly_results = {}
for i, column in enumerate(error_data.columns):
    model = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")  # 设定异常率
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
plt.ylabel("Standardized Error")
plt.legend()
plt.title("Error Trends of Agents")
plt.show()

# 输出检测结果
print(anomaly_results)

# 当前：模型假设大部分数据点都是“正常的”，并学习它们的边界。偏离正常模式的点：如果某个 Agent 的误差数据超出了学习到的分布范围，就被判定为异常

# LSTM：用 LSTM 预测下一个时刻的误差，如果误差远离预测值，则可能是拜占庭节点。
# 多 Agent 关联检测：比较多个 Agent 的误差模式，找出与群体行为不符的节点。