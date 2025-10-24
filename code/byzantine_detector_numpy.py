"""
使用纯NumPy实现的拜占庭节点检测
不依赖sklearn或PyTorch，适合快速原型验证
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

# ==================== 数据加载 ====================

def load_data_from_csv(csv_file='vhat_difference_log.csv'):
    """从CSV文件加载数据"""
    print(f"从 {csv_file} 加载数据...")

    time_data = []
    agent_data = [[] for _ in range(8)]

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            time_data.append(float(row[0]))
            for i in range(8):
                agent_data[i].append(float(row[i + 1]))

    time_data = np.array(time_data)
    for i in range(8):
        agent_data[i] = np.array(agent_data[i])

    print(f"✓ 加载完成: {len(time_data)} 个时间点, 8个智能体")
    return time_data, agent_data


# ==================== 特征提取 ====================

def extract_features(agent_time_series):
    """
    从单个智能体的时序数据中提取特征
    这些特征用于区分正常节点和拜占庭节点
    """

    # 1. 基本统计特征
    mean_val = np.mean(agent_time_series)
    std_val = np.std(agent_time_series)
    max_val = np.max(agent_time_series)
    min_val = np.min(agent_time_series)
    median_val = np.median(agent_time_series)

    # 2. 分位数特征
    q25 = np.percentile(agent_time_series, 25)
    q75 = np.percentile(agent_time_series, 75)
    iqr = q75 - q25
    range_val = max_val - min_val

    # 3. 时序特征
    diff = np.diff(agent_time_series)
    mean_change = np.mean(np.abs(diff))
    std_change = np.std(diff)
    max_change = np.max(np.abs(diff))

    # 4. 稳定性特征
    # 计算后半段的标准差（正常节点应该收敛，拜占庭节点持续波动）
    mid_point = len(agent_time_series) // 2
    late_std = np.std(agent_time_series[mid_point:])
    early_std = np.std(agent_time_series[:mid_point])
    std_ratio = late_std / (early_std + 1e-10)

    # 5. 异常点统计
    mean_threshold = mean_val + 2 * std_val
    num_outliers = np.sum(agent_time_series > mean_threshold)

    # 6. 收敛特征
    # 正常节点应该收敛，最后几个时间步的方差应该很小
    last_100 = agent_time_series[-100:]
    late_variance = np.var(last_100)

    features = np.array([
        mean_val,           # 0
        std_val,            # 1
        max_val,            # 2
        min_val,            # 3
        median_val,         # 4
        iqr,                # 5
        range_val,          # 6
        mean_change,        # 7
        std_change,         # 8
        max_change,         # 9
        std_ratio,          # 10
        num_outliers,       # 11
        late_variance,      # 12
        late_std            # 13
    ])

    return features


def extract_all_features(agent_data_list):
    """提取所有智能体的特征"""
    print("\n提取特征...")

    features = []
    for i, agent_data in enumerate(agent_data_list):
        feat = extract_features(agent_data)
        features.append(feat)
        print(f"  Agent {i}: 特征向量维度 = {len(feat)}")

    features = np.array(features)
    print(f"✓ 特征矩阵形状: {features.shape}")

    return features


# ==================== 检测方法 ====================

def detect_by_statistical_threshold(features, agent_data_list, known_byzantine=None):
    """
    方法1: 基于统计阈值的检测
    使用多个统计指标的组合来识别异常

    参数:
        features: (n_agents, n_features) 特征矩阵
        agent_data_list: 原始时序数据列表
        known_byzantine: 已知的拜占庭节点ID（用于可视化）
    """
    print("\n" + "="*60)
    print("方法1: 基于统计阈值的检测")
    print("="*60)

    n_agents = features.shape[0]

    # 关键特征索引
    # 拜占庭节点特征：高标准差、大变化率、未收敛
    std_idx = 1
    mean_change_idx = 7
    late_variance_idx = 12

    # 计算阈值（使用中位数绝对偏差MAD，对异常值更robust）
    def mad_threshold(data, n_sigma=3):
        """使用MAD计算阈值"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        threshold = median + n_sigma * 1.4826 * mad  # 1.4826是使MAD与标准差一致的系数
        return threshold

    # 对每个特征计算阈值
    std_threshold = mad_threshold(features[:, std_idx], n_sigma=2)
    change_threshold = mad_threshold(features[:, mean_change_idx], n_sigma=2)
    variance_threshold = mad_threshold(features[:, late_variance_idx], n_sigma=2)

    print(f"检测阈值:")
    print(f"  - 标准差阈值: {std_threshold:.6f}")
    print(f"  - 平均变化率阈值: {change_threshold:.6f}")
    print(f"  - 后期方差阈值: {variance_threshold:.6f}")

    # 异常评分（多个指标的组合）
    anomaly_scores = np.zeros(n_agents)

    for i in range(n_agents):
        score = 0

        # 标准差异常高
        if features[i, std_idx] > std_threshold:
            score += 1

        # 变化率异常大
        if features[i, mean_change_idx] > change_threshold:
            score += 1

        # 未收敛（后期方差仍然很大）
        if features[i, late_variance_idx] > variance_threshold:
            score += 1

        anomaly_scores[i] = score

    # 预测：评分>=2的判定为拜占庭节点
    predictions = (anomaly_scores >= 2).astype(int)

    print(f"\n异常评分:")
    for i in range(n_agents):
        status = "(Byzantine)" if (known_byzantine is not None and i == known_byzantine) else ""
        pred_label = "拜占庭" if predictions[i] == 1 else "正常"
        print(f"  Agent {i}: 评分={anomaly_scores[i]}, 预测={pred_label} {status}")

    return predictions, anomaly_scores


def detect_by_clustering(features, known_byzantine=None):
    """
    方法2: 基于K-means聚类的检测
    假设正常节点聚成一类，拜占庭节点是离群的

    参数:
        features: (n_agents, n_features) 特征矩阵
        known_byzantine: 已知的拜占庭节点ID（用于可视化）
    """
    print("\n" + "="*60)
    print("方法2: 基于K-means聚类的检测")
    print("="*60)

    n_agents, n_features = features.shape

    # 归一化特征
    features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

    # 简单的K-means实现（k=2）
    def kmeans(X, k=2, max_iters=100):
        """简单的K-means实现"""
        # 随机初始化中心
        np.random.seed(42)
        indices = np.random.choice(len(X), k, replace=False)
        centers = X[indices]

        for _ in range(max_iters):
            # 分配样本到最近的中心
            distances = np.array([[np.linalg.norm(x - center) for center in centers] for x in X])
            labels = np.argmin(distances, axis=1)

            # 更新中心
            new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
                                   for i in range(k)])

            # 检查收敛
            if np.allclose(centers, new_centers):
                break

            centers = new_centers

        return labels, centers

    labels, centers = kmeans(features_normalized, k=2)

    # 判断哪个聚类是拜占庭节点
    # 通常拜占庭节点的特征值更极端（标准差、变化率更大）
    cluster_means = np.array([features[labels == i].mean(axis=0) for i in range(2)])

    # 使用标准差和变化率的平均值来判断
    std_idx = 1
    change_idx = 7
    cluster_0_score = cluster_means[0, std_idx] + cluster_means[0, change_idx]
    cluster_1_score = cluster_means[1, std_idx] + cluster_means[1, change_idx]

    byzantine_cluster = 0 if cluster_0_score > cluster_1_score else 1

    predictions = (labels == byzantine_cluster).astype(int)

    print(f"聚类结果:")
    print(f"  - 聚类0大小: {np.sum(labels == 0)}")
    print(f"  - 聚类1大小: {np.sum(labels == 1)}")
    print(f"  - 拜占庭聚类判定: 聚类{byzantine_cluster}")

    print(f"\n预测结果:")
    for i in range(n_agents):
        status = "(Byzantine)" if (known_byzantine is not None and i == known_byzantine) else ""
        pred_label = "拜占庭" if predictions[i] == 1 else "正常"
        print(f"  Agent {i}: 聚类={labels[i]}, 预测={pred_label} {status}")

    return predictions, labels


def detect_by_distance(features, known_byzantine=None):
    """
    方法3: 基于距离的异常检测
    计算每个节点到群体中心的距离，距离最远的是拜占庭节点
    """
    print("\n" + "="*60)
    print("方法3: 基于马氏距离的异常检测")
    print("="*60)

    # 归一化特征
    features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

    # 计算均值和协方差
    mean = features_normalized.mean(axis=0)
    cov = np.cov(features_normalized.T)

    # 处理奇异协方差矩阵（添加小的对角扰动）
    cov_reg = cov + np.eye(cov.shape[0]) * 1e-6

    # 计算马氏距离
    try:
        cov_inv = np.linalg.inv(cov_reg)
        distances = np.array([
            np.sqrt((x - mean).T @ cov_inv @ (x - mean))
            for x in features_normalized
        ])
    except:
        # 如果协方差矩阵仍然奇异，使用欧氏距离
        print("  (使用欧氏距离代替马氏距离)")
        distances = np.array([np.linalg.norm(x - mean) for x in features_normalized])

    # 使用阈值检测异常
    threshold = np.mean(distances) + 2 * np.std(distances)

    predictions = (distances > threshold).astype(int)

    print(f"距离统计:")
    print(f"  - 平均距离: {np.mean(distances):.4f}")
    print(f"  - 标准差: {np.std(distances):.4f}")
    print(f"  - 检测阈值: {threshold:.4f}")

    print(f"\n预测结果:")
    for i in range(len(features)):
        status = "(Byzantine)" if (known_byzantine is not None and i == known_byzantine) else ""
        pred_label = "拜占庭" if predictions[i] == 1 else "正常"
        print(f"  Agent {i}: 距离={distances[i]:.4f}, 预测={pred_label} {status}")

    return predictions, distances


# ==================== 评估 ====================

def evaluate_predictions(predictions, true_byzantine):
    """评估预测结果"""

    n_agents = len(predictions)
    true_labels = np.zeros(n_agents)
    true_labels[true_byzantine] = 1

    # 计算指标
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))

    accuracy = (tp + tn) / n_agents
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n性能指标:")
    print(f"  - 准确率 (Accuracy): {accuracy:.4f}")
    print(f"  - 精确率 (Precision): {precision:.4f}")
    print(f"  - 召回率 (Recall): {recall:.4f}")
    print(f"  - F1分数: {f1:.4f}")
    print(f"\n混淆矩阵:")
    print(f"  真正常/预测正常 (TN): {tn}")
    print(f"  真正常/预测拜占庭 (FP): {fp}")
    print(f"  真拜占庭/预测正常 (FN): {fn}")
    print(f"  真拜占庭/预测拜占庭 (TP): {tp}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': [[tn, fp], [fn, tp]]
    }


# ==================== 可视化 ====================

def visualize_results(time_data, agent_data_list, features, predictions_dict,
                     true_byzantine, metrics_dict):
    """可视化检测结果"""

    n_methods = len(predictions_dict)
    method_names = list(predictions_dict.keys())

    # 创建大图
    fig = plt.figure(figsize=(18, 12))

    # 1. 原始时序数据（所有智能体）
    ax1 = plt.subplot(3, 3, 1)
    colors = plt.cm.tab10(np.arange(8))
    for i, agent_data in enumerate(agent_data_list):
        label = f'Agent {i}'
        if i == true_byzantine:
            label += ' (Byzantine)'
            ax1.plot(time_data, agent_data, linewidth=2, color=colors[i],
                    linestyle='--', label=label)
        else:
            ax1.plot(time_data, agent_data, linewidth=1, color=colors[i],
                    alpha=0.7, label=label if i < 3 else '')

    ax1.set_title('Observer Consensus Error (v_hat差异)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('||v_hat - filtered_mean||')
    ax1.set_yscale('log')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. 特征分布（前3个关键特征）
    feature_names = ['标准差', '平均变化率', '后期方差']
    feature_indices = [1, 7, 12]

    for idx, (feat_name, feat_idx) in enumerate(zip(feature_names, feature_indices)):
        ax = plt.subplot(3, 3, 2 + idx)

        # 正常节点
        normal_features = [features[i, feat_idx] for i in range(8) if i != true_byzantine]
        # 拜占庭节点
        byz_feature = features[true_byzantine, feat_idx]

        ax.bar(range(len(normal_features)), normal_features, color='steelblue',
              alpha=0.7, label='Normal')
        ax.bar([len(normal_features)], [byz_feature], color='red',
              alpha=0.7, label='Byzantine')

        ax.set_title(f'特征对比: {feat_name}')
        ax.set_xlabel('Agent')
        ax.set_ylabel(feat_name)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

    # 3. 每个方法的预测结果和性能
    for idx, method_name in enumerate(method_names):
        # 预测结果可视化
        ax = plt.subplot(3, 3, 5 + idx)

        predictions = predictions_dict[method_name]
        colors_pred = ['green' if pred == 0 else 'red' for pred in predictions]

        # 标记真实的拜占庭节点
        sizes = [200 if i == true_byzantine else 100 for i in range(8)]

        ax.scatter(range(8), predictions, c=colors_pred, s=sizes,
                  alpha=0.7, edgecolors='black', linewidths=2)

        ax.set_title(f'{method_name} - 预测结果')
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('预测标签 (0=正常, 1=拜占庭)')
        ax.set_ylim([-0.5, 1.5])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['正常', '拜占庭'])
        ax.grid(True, axis='y', alpha=0.3)

        # 添加注释
        ax.axvline(true_byzantine, color='orange', linestyle=':', alpha=0.5,
                  label='True Byzantine')
        ax.legend(fontsize=8)

    # 4. 性能对比
    ax_perf = plt.subplot(3, 3, 8)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.25

    for idx, method_name in enumerate(method_names):
        values = [
            metrics_dict[method_name]['accuracy'],
            metrics_dict[method_name]['precision'],
            metrics_dict[method_name]['recall'],
            metrics_dict[method_name]['f1']
        ]
        ax_perf.bar(x + idx * width, values, width, label=method_name, alpha=0.8)

    ax_perf.set_title('方法性能对比')
    ax_perf.set_ylabel('Score')
    ax_perf.set_xticks(x + width)
    ax_perf.set_xticklabels(metrics, rotation=15, ha='right')
    ax_perf.set_ylim([0, 1.1])
    ax_perf.legend(fontsize=8)
    ax_perf.grid(True, axis='y', alpha=0.3)

    # 5. 特征重要性（基于方差贡献）
    ax_feat = plt.subplot(3, 3, 9)

    # 计算每个特征的区分能力（拜占庭 vs 正常的差异）
    normal_mean = np.mean([features[i] for i in range(8) if i != true_byzantine], axis=0)
    byz_value = features[true_byzantine]
    discrimination = np.abs(byz_value - normal_mean) / (normal_mean + 1e-10)

    all_feature_names = ['均值', '标准差', '最大值', '最小值', '中位数',
                        'IQR', '范围', '平均变化率', '变化率标准差', '最大变化',
                        '标准差比', '异常点数', '后期方差', '后期标准差']

    # 只显示top 8个特征
    top_indices = np.argsort(discrimination)[::-1][:8]
    top_names = [all_feature_names[i] for i in top_indices]
    top_scores = discrimination[top_indices]

    ax_feat.barh(range(len(top_scores)), top_scores, color='teal', alpha=0.7)
    ax_feat.set_yticks(range(len(top_scores)))
    ax_feat.set_yticklabels(top_names)
    ax_feat.set_xlabel('区分能力（归一化差异）')
    ax_feat.set_title('Top 8 区分特征')
    ax_feat.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('byzantine_detection_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ 结果图已保存至 byzantine_detection_results.png")


# ==================== 主程序 ====================

if __name__ == '__main__':
    print("="*60)
    print("拜占庭节点检测 - 纯NumPy实现")
    print("="*60)

    # 加载数据
    time_data, agent_data_list = load_data_from_csv('vhat_difference_log.csv')

    # 已知的拜占庭节点（从原代码得知是Agent 4）
    true_byzantine = 4

    # 提取特征
    features = extract_all_features(agent_data_list)

    # 运行三种检测方法
    predictions_dict = {}
    metrics_dict = {}

    # 方法1: 统计阈值
    pred1, scores1 = detect_by_statistical_threshold(features, agent_data_list, true_byzantine)
    predictions_dict['Statistical Threshold'] = pred1
    metrics_dict['Statistical Threshold'] = evaluate_predictions(pred1, true_byzantine)

    # 方法2: 聚类
    pred2, labels2 = detect_by_clustering(features, true_byzantine)
    predictions_dict['K-means Clustering'] = pred2
    metrics_dict['K-means Clustering'] = evaluate_predictions(pred2, true_byzantine)

    # 方法3: 距离
    pred3, distances3 = detect_by_distance(features, true_byzantine)
    predictions_dict['Distance-based'] = pred3
    metrics_dict['Distance-based'] = evaluate_predictions(pred3, true_byzantine)

    # 可视化
    print("\n" + "="*60)
    print("生成可视化结果")
    print("="*60)

    visualize_results(time_data, agent_data_list, features, predictions_dict,
                     true_byzantine, metrics_dict)

    # 总结
    print("\n" + "="*60)
    print("检测总结")
    print("="*60)

    print(f"\n真实拜占庭节点: Agent {true_byzantine}")
    print(f"\n各方法检测结果:")

    for method_name, predictions in predictions_dict.items():
        detected = [i for i, pred in enumerate(predictions) if pred == 1]
        correct = true_byzantine in detected
        status = "✓" if correct else "✗"
        print(f"  {status} {method_name}: 检测到 {detected}")

    print(f"\n最佳方法: ", end='')
    best_method = max(metrics_dict.items(), key=lambda x: x[1]['f1'])
    print(f"{best_method[0]} (F1={best_method[1]['f1']:.4f})")

    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
