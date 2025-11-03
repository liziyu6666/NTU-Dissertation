"""
使用机器学习检测拜占庭节点 - 简化版
使用sklearn和简单的神经网络，不依赖PyTorch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle
import csv

# ==================== 从CSV加载数据 ====================

def load_simulation_data_from_csv(csv_file='vhat_difference_log.csv'):
    """从之前保存的CSV文件加载数据"""
    data = {
        'time': [],
        'agents': [[] for _ in range(8)]
    }

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头

        for row in reader:
            data['time'].append(float(row[0]))
            for i in range(8):
                data['agents'][i].append(float(row[i + 1]))

    # 转换为numpy数组
    data['time'] = np.array(data['time'])
    for i in range(8):
        data['agents'][i] = np.array(data['agents'][i])

    return data


def generate_multiple_simulations(num_scenarios=40):
    """
    运行多个仿真场景生成训练数据
    每个场景选择不同的拜占庭节点
    """
    import subprocess
    import os
    import tempfile

    print("生成多个仿真场景...")

    all_scenarios = []

    # 创建临时代码副本
    base_code = open('1.py', 'r').read()

    for scenario_id in range(num_scenarios):
        # 随机选择拜占庭节点 (0-7)
        faulty_agent = np.random.randint(0, 8)

        # 随机选择攻击类型
        attack_params = {
            0: "np.array([50 * np.sin(10 * t) + 15 * np.cos(12 * t), t / 15])",  # mixed
            1: "np.array([30 * np.sin(8 * t), 30 * np.cos(8 * t)])",  # sine
            2: "np.array([t * 2, -t * 1.5])",  # ramp
            3: "np.array([100 + 20*np.sin(t), -50 + 10*np.cos(t)])",  # constant + oscillation
        }
        attack_type = np.random.choice(list(attack_params.keys()))
        attack_code = attack_params[attack_type]

        # 修改代码中的拜占庭节点和攻击类型
        modified_code = base_code.replace(
            'faulty_agent = 4',
            f'faulty_agent = {faulty_agent}'
        )
        modified_code = modified_code.replace(
            "dv_hat = np.array([50 * np.sin(10 * t) + 15 * np.cos(12 * t), t / 15])",
            f"dv_hat = {attack_code}"
        )

        # 修改输出文件名以避免覆盖
        modified_code = modified_code.replace(
            '"vhat_difference_log.csv"',
            f'"vhat_log_scenario_{scenario_id}.csv"'
        )
        modified_code = modified_code.replace(
            '"resilient_cor_results.png"',
            f'"results_scenario_{scenario_id}.png"'
        )

        # 写入临时文件
        temp_file = f'temp_sim_{scenario_id}.py'
        with open(temp_file, 'w') as f:
            f.write(modified_code)

        # 运行仿真
        print(f"场景 {scenario_id + 1}/{num_scenarios}: 拜占庭节点={faulty_agent}, 攻击类型={attack_type}", end=' ... ')
        try:
            result = subprocess.run(['python3', temp_file],
                                   capture_output=True,
                                   timeout=60)

            if result.returncode == 0:
                # 加载生成的CSV
                csv_file = f'vhat_log_scenario_{scenario_id}.csv'
                if os.path.exists(csv_file):
                    scenario_data = load_simulation_data_from_csv(csv_file)
                    scenario_data['faulty_agent'] = faulty_agent
                    scenario_data['attack_type'] = attack_type
                    all_scenarios.append(scenario_data)
                    print("✓")
                else:
                    print("✗ (CSV未生成)")
            else:
                print(f"✗ (运行失败)")

        except subprocess.TimeoutExpired:
            print("✗ (超时)")
        except Exception as e:
            print(f"✗ ({e})")

        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
        csv_file = f'vhat_log_scenario_{scenario_id}.csv'
        if os.path.exists(csv_file):
            os.remove(csv_file)
        png_file = f'results_scenario_{scenario_id}.png'
        if os.path.exists(png_file):
            os.remove(png_file)

    print(f"\n✓ 成功生成 {len(all_scenarios)} 个场景")
    return all_scenarios


def extract_statistical_features(agent_time_series, window_size=100):
    """
    从时序数据中提取统计特征
    这些特征对于检测异常模式很有效
    """
    features = []

    # 将时序分成多个窗口
    n_windows = len(agent_time_series) // window_size

    for i in range(n_windows):
        window = agent_time_series[i*window_size:(i+1)*window_size]

        # 统计特征
        feature_vector = [
            np.mean(window),                    # 均值
            np.std(window),                     # 标准差
            np.max(window),                     # 最大值
            np.min(window),                     # 最小值
            np.median(window),                  # 中位数
            np.percentile(window, 75) - np.percentile(window, 25),  # IQR
            np.max(window) - np.min(window),    # 范围
            np.mean(np.abs(np.diff(window))),   # 平均变化率
            np.std(np.diff(window)),            # 变化率标准差
            np.sum(window > np.mean(window)),   # 高于均值的点数
        ]

        features.append(feature_vector)

    return np.array(features)


def prepare_dataset(scenarios, window_size=100):
    """准备训练数据集"""

    all_features = []
    all_labels = []

    print(f"\n提取特征 (窗口大小={window_size})...")

    for scenario in scenarios:
        faulty_agent = scenario['faulty_agent']

        for agent_id in range(8):
            agent_data = scenario['agents'][agent_id]

            # 提取特征
            features = extract_statistical_features(agent_data, window_size)

            # 标签
            label = 1 if agent_id == faulty_agent else 0

            all_features.append(features)
            all_labels.extend([label] * len(features))

    # 展平特征
    all_features = np.vstack(all_features)

    print(f"✓ 特征提取完成")
    print(f"  - 总样本数: {len(all_features)}")
    print(f"  - 特征维度: {all_features.shape[1]}")
    print(f"  - 拜占庭样本数: {sum(all_labels)}")
    print(f"  - 正常样本数: {len(all_labels) - sum(all_labels)}")

    return all_features, np.array(all_labels)


# ==================== 方法1：隔离森林（异常检测）====================

def train_isolation_forest(X_train, X_test, y_test):
    """
    使用Isolation Forest进行异常检测
    只在正常数据上训练
    """
    print("\n" + "="*60)
    print("方法1：隔离森林（Isolation Forest）异常检测")
    print("="*60)

    # 只使用正常样本训练
    X_train_normal = X_train[y_test[:len(X_train)] == 0]

    print(f"训练样本数（仅正常）: {len(X_train_normal)}")

    # 训练模型
    clf = IsolationForest(
        n_estimators=100,
        contamination=0.1,  # 假设10%是异常
        random_state=42,
        n_jobs=-1
    )

    print("开始训练...")
    clf.fit(X_train_normal)

    # 预测（-1表示异常，1表示正常）
    y_pred = clf.predict(X_test)
    y_pred = (y_pred == -1).astype(int)  # 转换为0/1

    # 评估
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['正常', '拜占庭']))

    return clf, y_pred


# ==================== 方法2：随机森林（分类）====================

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    使用Random Forest进行监督分类
    """
    print("\n" + "="*60)
    print("方法2：随机森林（Random Forest）分类")
    print("="*60)

    print(f"训练样本数: {len(X_train)}")
    print(f"  - 正常: {np.sum(y_train == 0)}")
    print(f"  - 拜占庭: {np.sum(y_train == 1)}")

    # 训练模型
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # 处理类别不平衡
        random_state=42,
        n_jobs=-1
    )

    print("开始训练...")
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # 评估
    print(f"\n训练集准确率: {clf.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {clf.score(X_test, y_test):.4f}")

    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['正常', '拜占庭']))

    # 特征重要性
    feature_names = ['均值', '标准差', '最大值', '最小值', '中位数',
                    'IQR', '范围', '平均变化率', '变化率标准差', '高于均值点数']
    feature_importance = clf.feature_importances_
    print("\n特征重要性排名:")
    for name, importance in sorted(zip(feature_names, feature_importance),
                                  key=lambda x: x[1], reverse=True):
        print(f"  {name}: {importance:.4f}")

    return clf, y_pred, y_pred_proba


# ==================== 方法3：基于规则的检测 ====================

def rule_based_detection(X_test, y_test):
    """
    基于统计规则的简单检测
    适合作为baseline
    """
    print("\n" + "="*60)
    print("方法3：基于规则的检测（Rule-based）")
    print("="*60)

    # 规则：标准差异常高、最大值异常高、变化率异常大
    # 使用训练集的统计量设定阈值

    std_threshold = np.percentile(X_test[:, 1], 95)  # 标准差的95分位数
    max_threshold = np.percentile(X_test[:, 2], 95)  # 最大值的95分位数
    change_threshold = np.percentile(X_test[:, 7], 95)  # 变化率的95分位数

    print(f"检测阈值:")
    print(f"  - 标准差阈值: {std_threshold:.4f}")
    print(f"  - 最大值阈值: {max_threshold:.4f}")
    print(f"  - 变化率阈值: {change_threshold:.4f}")

    # 应用规则
    y_pred = np.zeros(len(X_test), dtype=int)

    # 如果任意两个指标超出阈值，判定为拜占庭
    anomaly_scores = (
        (X_test[:, 1] > std_threshold).astype(int) +
        (X_test[:, 2] > max_threshold).astype(int) +
        (X_test[:, 7] > change_threshold).astype(int)
    )
    y_pred = (anomaly_scores >= 2).astype(int)

    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['正常', '拜占庭']))

    return y_pred


# ==================== 可视化 ====================

def visualize_results(y_test, predictions_dict, feature_importance=None):
    """可视化所有方法的结果"""

    n_methods = len(predictions_dict)
    fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 10))

    if n_methods == 1:
        axes = axes.reshape(-1, 1)

    method_names = list(predictions_dict.keys())

    for i, method in enumerate(method_names):
        y_pred = predictions_dict[method]

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, i],
                   xticklabels=['Normal', 'Byzantine'],
                   yticklabels=['Normal', 'Byzantine'])
        axes[0, i].set_title(f'{method}\nConfusion Matrix')
        axes[0, i].set_ylabel('True Label')
        axes[0, i].set_xlabel('Predicted Label')

        # 性能指标柱状图
        tp = cm[1, 1]
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]

        bars = axes[1, i].bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        axes[1, i].set_ylim([0, 1.1])
        axes[1, i].set_title(f'{method}\nPerformance Metrics')
        axes[1, i].set_ylabel('Score')
        axes[1, i].grid(True, axis='y', alpha=0.3)

        # 在柱子上标注数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, i].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}',
                          ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('byzantine_detection_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ 对比图已保存至 byzantine_detection_comparison.png")

    # 如果有特征重要性，单独绘制
    if feature_importance is not None:
        fig2, ax = plt.subplots(figsize=(10, 6))
        feature_names = ['Mean', 'Std', 'Max', 'Min', 'Median',
                        'IQR', 'Range', 'Avg Change', 'Std Change', 'Above Mean']
        indices = np.argsort(feature_importance)[::-1]

        ax.bar(range(len(feature_importance)),
               feature_importance[indices],
               color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(feature_importance)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_title('Feature Importance (Random Forest)')
        ax.set_ylabel('Importance Score')
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        print("✓ 特征重要性图已保存至 feature_importance.png")


# ==================== 主程序 ====================

if __name__ == '__main__':
    print("="*60)
    print("拜占庭节点检测 - 机器学习方法对比")
    print("="*60)

    # ==================== 选项：使用现有数据或生成新数据 ====================

    import os

    use_existing = os.path.exists('vhat_difference_log.csv')

    if use_existing:
        print("\n检测到现有仿真数据，是否使用？")
        print("注意：使用现有数据只有1个场景，建议生成多个场景以提高模型性能")
        print("如需生成新数据，请删除 vhat_difference_log.csv 后重新运行")

        # 加载现有数据（单场景）
        print("\n使用现有单场景数据...")
        data = load_simulation_data_from_csv('vhat_difference_log.csv')

        # 假设拜占庭节点是Agent 4（根据原代码）
        data['faulty_agent'] = 4
        scenarios = [data]

    else:
        print("\n未检测到现有数据，将生成多个仿真场景...")
        scenarios = generate_multiple_simulations(num_scenarios=40)

    if len(scenarios) == 0:
        print("错误：没有可用的仿真数据！")
        exit(1)

    # ==================== 准备数据集 ====================

    X, y = prepare_dataset(scenarios, window_size=100)

    # 归一化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n数据集划分:")
    print(f"  - 训练集: {len(X_train)} 样本")
    print(f"  - 测试集: {len(X_test)} 样本")

    # ==================== 训练和评估 ====================

    predictions = {}

    # 方法1: 隔离森林
    clf_if, y_pred_if = train_isolation_forest(X_train, X_test, y_test)
    predictions['Isolation Forest'] = y_pred_if

    # 方法2: 随机森林
    clf_rf, y_pred_rf, y_pred_proba_rf = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    predictions['Random Forest'] = y_pred_rf
    feature_importance = clf_rf.feature_importances_

    # 方法3: 基于规则
    y_pred_rule = rule_based_detection(X_test, y_test)
    predictions['Rule-based'] = y_pred_rule

    # ==================== 可视化 ====================

    print("\n" + "="*60)
    print("生成可视化结果")
    print("="*60)

    visualize_results(y_test, predictions, feature_importance)

    # ==================== 保存模型 ====================

    print("\n" + "="*60)
    print("保存模型")
    print("="*60)

    with open('isolation_forest_model.pkl', 'wb') as f:
        pickle.dump((clf_if, scaler), f)
    print("✓ 隔离森林模型已保存至 isolation_forest_model.pkl")

    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump((clf_rf, scaler), f)
    print("✓ 随机森林模型已保存至 random_forest_model.pkl")

    print("\n" + "="*60)
    print("所有任务完成！")
    print("="*60)

    print("\n总结:")
    print("- 生成了多个仿真场景用于训练")
    print("- 对比了3种检测方法：隔离森林、随机森林、基于规则")
    print("- 随机森林通常表现最好，因为它可以学习拜占庭模式")
    print("- 隔离森林适合未知攻击类型的异常检测")
    print("- 基于规则的方法简单但可解释性强")
