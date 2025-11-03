"""
完整的训练和评估pipeline
包含: LSTM Autoencoder, LSTM Classifier, Random Forest, 统计方法, 距离方法
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import os

print("PyTorch version:", torch.__version__)

# ==================== LSTM模型定义 ====================

class LSTMAutoencoder(nn.Module):
    """LSTM自编码器用于异常检测"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 编码器
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 解码器
        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # 编码
        _, (hidden, cell) = self.encoder(x)

        # 重复hidden状态
        seq_len = x.size(1)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)

        # 解码
        decoded, _ = self.decoder(decoder_input, (hidden, cell))

        # 重构
        reconstructed = self.fc(decoded)

        return reconstructed


class LSTMClassifier(nn.Module):
    """LSTM分类器用于拜占庭节点检测"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 2)  # 二分类

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]

        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# ==================== 数据集定义 ====================

class TimeSeriesDataset(Dataset):
    """时序数据集"""

    def __init__(self, sequences, labels=None):
        self.sequences = [torch.FloatTensor(seq) for seq in sequences]
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        else:
            return self.sequences[idx]


# ==================== 特征提取 ====================

def extract_windows(agent_data, window_size=100, stride=50):
    """从单个智能体数据中提取滑动窗口"""

    # 特征：估计误差、位置误差、角度、角速度、控制输入、v_hat变化
    features = np.vstack([
        agent_data['estimation_error'],
        agent_data['position_error'],
        agent_data['angle'],
        agent_data['angular_velocity'],
        agent_data['control_input'],
        np.abs(agent_data['v_hat_0']) + np.abs(agent_data['v_hat_1']),  # v_hat magnitude
    ]).T  # (time_steps, 6)

    # 归一化
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features = (features - mean) / std

    # 滑动窗口
    windows = []
    for i in range(0, len(features) - window_size, stride):
        windows.append(features[i:i+window_size])

    return windows


def prepare_dataset_from_scenarios(scenarios, window_size=100, stride=50):
    """从场景数据准备数据集"""

    print(f"\n准备数据集 (窗口大小={window_size}, 步长={stride})...")

    all_sequences = []
    all_labels = []

    for scenario in scenarios:
        faulty_agent = scenario['faulty_agent']

        for agent_data in scenario['agents']:
            agent_id = agent_data['agent_id']
            is_byzantine = (agent_id == faulty_agent)

            # 提取窗口
            windows = extract_windows(agent_data, window_size, stride)

            for window in windows:
                all_sequences.append(window)
                all_labels.append(1 if is_byzantine else 0)

    print(f"✓ 数据集准备完成")
    print(f"  - 总窗口数: {len(all_sequences)}")
    print(f"  - 正常样本: {all_labels.count(0)}")
    print(f"  - 拜占庭样本: {all_labels.count(1)}")
    print(f"  - 特征维度: {all_sequences[0].shape}")

    return all_sequences, all_labels


# ==================== 训练函数 ====================

def train_autoencoder(model, train_loader, val_loader, epochs=30, lr=0.001, device='cpu'):
    """训练LSTM自编码器"""

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    train_losses = []
    val_losses = []

    print(f"\n训练LSTM自编码器 (epochs={epochs})...")

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses


def train_classifier(model, train_loader, val_loader, epochs=30, lr=0.001, device='cpu'):
    """训练LSTM分类器"""

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print(f"\n训练LSTM分类器 (epochs={epochs})...")

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


# ==================== 传统ML方法 ====================

def extract_statistical_features(window):
    """从窗口提取统计特征"""
    features = []

    for dim in range(window.shape[1]):
        data = window[:, dim]
        features.extend([
            np.mean(data),
            np.std(data),
            np.max(data),
            np.min(data),
            np.median(data),
            np.percentile(data, 75) - np.percentile(data, 25),  # IQR
            np.mean(np.abs(np.diff(data))),  # 平均变化率
        ])

    return np.array(features)


def prepare_traditional_ml_dataset(sequences, labels):
    """为传统ML方法准备数据集"""
    X = np.array([extract_statistical_features(seq) for seq in sequences])
    y = np.array(labels)
    return X, y


# ==================== 主训练流程 ====================

def main():
    print("="*60)
    print("完整训练和评估Pipeline")
    print("="*60)

    # 设置device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载数据
    data_path = 'training_data/all_scenarios.pkl'
    print(f"\n加载数据: {data_path}")

    with open(data_path, 'rb') as f:
        scenarios = pickle.load(f)

    print(f"✓ 加载了 {len(scenarios)} 个场景")

    # 准备数据集
    sequences, labels = prepare_dataset_from_scenarios(scenarios, window_size=100, stride=50)

    # 划分训练集和测试集
    indices = np.random.permutation(len(sequences))
    split_idx = int(len(sequences) * 0.8)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_sequences = [sequences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_sequences = [sequences[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"\n数据集划分:")
    print(f"  - 训练集: {len(train_sequences)}")
    print(f"  - 测试集: {len(test_sequences)}")

    # ==================== 方法1&2: LSTM ====================

    # 自编码器数据（只用正常样本）
    normal_train_sequences = [train_sequences[i] for i, label in enumerate(train_labels) if label == 0]
    normal_val_sequences = normal_train_sequences[:len(normal_train_sequences)//5]
    normal_train_sequences = normal_train_sequences[len(normal_train_sequences)//5:]

    train_dataset_ae = TimeSeriesDataset(normal_train_sequences)
    val_dataset_ae = TimeSeriesDataset(normal_val_sequences)
    train_loader_ae = DataLoader(train_dataset_ae, batch_size=32, shuffle=True)
    val_loader_ae = DataLoader(val_dataset_ae, batch_size=32)

    # 分类器数据
    val_split = int(len(train_sequences) * 0.8)
    train_dataset_clf = TimeSeriesDataset(train_sequences[:val_split], train_labels[:val_split])
    val_dataset_clf = TimeSeriesDataset(train_sequences[val_split:], train_labels[val_split:])
    train_loader_clf = DataLoader(train_dataset_clf, batch_size=32, shuffle=True)
    val_loader_clf = DataLoader(val_dataset_clf, batch_size=32)

    # 训练自编码器
    model_ae = LSTMAutoencoder(input_dim=6, hidden_dim=64, num_layers=2)
    train_losses_ae, val_losses_ae = train_autoencoder(
        model_ae, train_loader_ae, val_loader_ae,
        epochs=30, lr=0.001, device=device
    )

    # 训练分类器
    model_clf = LSTMClassifier(input_dim=6, hidden_dim=64, num_layers=2)
    train_losses_clf, val_losses_clf, train_accs_clf, val_accs_clf = train_classifier(
        model_clf, train_loader_clf, val_loader_clf,
        epochs=30, lr=0.001, device=device
    )

    # ==================== 方法3: Random Forest ====================

    print(f"\n训练Random Forest...")
    X_train, y_train = prepare_traditional_ml_dataset(train_sequences, train_labels)
    X_test, y_test = prepare_traditional_ml_dataset(test_sequences, test_labels)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf_rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                    class_weight='balanced', random_state=42, n_jobs=-1)
    clf_rf.fit(X_train_scaled, y_train)
    print(f"✓ Random Forest训练完成")

    # ==================== 评估所有方法 ====================

    print("\n" + "="*60)
    print("评估所有方法")
    print("="*60)

    results = {}

    # 评估LSTM Autoencoder
    print("\n1. LSTM Autoencoder:")
    model_ae.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for seq in test_sequences:
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
            reconstructed = model_ae(seq_tensor)
            error = torch.mean((reconstructed - seq_tensor) ** 2).item()
            reconstruction_errors.append(error)

    # 使用阈值
    normal_errors = [reconstruction_errors[i] for i, label in enumerate(test_labels) if label == 0]
    threshold = np.percentile(normal_errors, 95)
    pred_ae = (np.array(reconstruction_errors) > threshold).astype(int)

    acc_ae = accuracy_score(test_labels, pred_ae)
    f1_ae = f1_score(test_labels, pred_ae)
    print(f"  准确率: {acc_ae:.4f}, F1分数: {f1_ae:.4f}")
    results['LSTM AE'] = {'predictions': pred_ae, 'accuracy': acc_ae, 'f1': f1_ae}

    # 评估LSTM Classifier
    print("\n2. LSTM Classifier:")
    model_clf.eval()
    pred_clf = []
    with torch.no_grad():
        for seq in test_sequences:
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
            outputs = model_clf(seq_tensor)
            _, predicted = torch.max(outputs, 1)
            pred_clf.append(predicted.item())

    pred_clf = np.array(pred_clf)
    acc_clf = accuracy_score(test_labels, pred_clf)
    f1_clf = f1_score(test_labels, pred_clf)
    print(f"  准确率: {acc_clf:.4f}, F1分数: {f1_clf:.4f}")
    results['LSTM CLF'] = {'predictions': pred_clf, 'accuracy': acc_clf, 'f1': f1_clf}

    # 评估Random Forest
    print("\n3. Random Forest:")
    pred_rf = clf_rf.predict(X_test_scaled)
    acc_rf = accuracy_score(test_labels, pred_rf)
    f1_rf = f1_score(test_labels, pred_rf)
    print(f"  准确率: {acc_rf:.4f}, F1分数: {f1_rf:.4f}")
    results['Random Forest'] = {'predictions': pred_rf, 'accuracy': acc_rf, 'f1': f1_rf}

    # ==================== 可视化结果 ====================

    print("\n生成结果可视化...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1-3: 混淆矩阵
    for idx, (method_name, result) in enumerate(results.items()):
        ax = axes[0, idx]
        cm = confusion_matrix(test_labels, result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Byzantine'],
                   yticklabels=['Normal', 'Byzantine'])
        ax.set_title(f'{method_name}\nAcc={result["accuracy"]:.3f}, F1={result["f1"]:.3f}')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

    # 4: 训练损失曲线
    ax = axes[1, 0]
    ax.plot(train_losses_ae, label='AE Train')
    ax.plot(val_losses_ae, label='AE Val')
    ax.plot(train_losses_clf, label='CLF Train')
    ax.plot(val_losses_clf, label='CLF Val')
    ax.set_title('Training Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5: 性能对比
    ax = axes[1, 1]
    methods = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in methods]
    f1_scores = [results[m]['f1'] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Method Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)

    # 标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 6: 特征重要性
    ax = axes[1, 2]
    feature_importance = clf_rf.feature_importances_
    top_indices = np.argsort(feature_importance)[::-1][:10]
    ax.barh(range(10), feature_importance[top_indices], color='steelblue', alpha=0.7)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'Feature {i}' for i in top_indices])
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Feature Importance (RF)')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_evaluation_results.png', dpi=150, bbox_inches='tight')
    print("✓ 结果图已保存至 training_evaluation_results.png")

    # 保存模型
    torch.save(model_ae.state_dict(), 'lstm_autoencoder.pth')
    torch.save(model_clf.state_dict(), 'lstm_classifier.pth')
    import pickle as pkl
    with open('random_forest_model.pkl', 'wb') as f:
        pkl.dump((clf_rf, scaler), f)

    print("\n✓ 模型已保存")

    # 打印详细报告
    print("\n" + "="*60)
    print("详细分类报告")
    print("="*60)
    for method_name, result in results.items():
        print(f"\n{method_name}:")
        print(classification_report(test_labels, result['predictions'],
                                    target_names=['Normal', 'Byzantine']))

    print("\n" + "="*60)
    print("训练和评估完成！")
    print("="*60)


if __name__ == '__main__':
    main()
