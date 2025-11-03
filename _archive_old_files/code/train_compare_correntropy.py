"""
对比实验：7维特征 vs 10维特征（含Correntropy）
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import time

torch.manual_seed(42)
np.random.seed(42)

# ================== 数据集类 ==================
class ByzantineDataset(Dataset):
    """支持可变维度特征的数据集"""

    def __init__(self, data_path, window_size=50, stride=50, feature_dims=7):
        self.window_size = window_size
        self.stride = stride
        self.feature_dims = feature_dims  # 7 or 10

        # 加载元数据
        with open(os.path.join(data_path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        print(f"加载 {len(metadata)} 个场景... (特征维度: {feature_dims})")

        self.windows = []
        self.labels = []

        for meta in metadata:
            filepath = os.path.join(data_path, meta['filename'])

            with open(filepath, 'rb') as f:
                scenario = pickle.load(f)

            byzantine_id = scenario['faulty_agent']

            for agent in scenario['agents']:
                agent_id = agent['agent_id']
                is_byzantine = (agent_id == byzantine_id)

                # 提取特征（根据维度）
                if feature_dims == 7:
                    # 原始7维特征
                    features = np.array([
                        agent['estimation_error'],
                        agent['position_error'],
                        agent['angle'],
                        agent['angular_velocity'],
                        agent['control_input'],
                        agent['v_hat_0'],
                        agent['v_hat_1']
                    ]).T
                elif feature_dims == 10:
                    # 10维特征（含Correntropy）
                    features = np.array([
                        agent['estimation_error'],
                        agent['position_error'],
                        agent['angle'],
                        agent['angular_velocity'],
                        agent['control_input'],
                        agent['v_hat_0'],
                        agent['v_hat_1'],
                        agent['avg_correntropy'],
                        agent['min_correntropy'],
                        agent['std_correntropy']
                    ]).T
                else:
                    raise ValueError(f"Unsupported feature_dims: {feature_dims}")

                # 归一化
                features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

                # 滑动窗口
                num_steps = len(features)
                for start in range(0, num_steps - self.window_size, self.stride):
                    window = features[start:start + self.window_size]
                    self.windows.append(window)
                    self.labels.append(1 if is_byzantine else 0)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        print(f"✓ 提取 {len(self.windows)} 个窗口")
        print(f"  - 窗口形状: {self.windows.shape}")
        print(f"  - Normal: {(self.labels == 0).sum()}, Byzantine: {(self.labels == 1).sum()}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]), torch.LongTensor([self.labels[idx]])[0]


# ================== LSTM模型 ==================
class LSTMBehaviorClassifier(nn.Module):
    """支持可变输入维度的LSTM分类器"""

    def __init__(self, input_dim=7, hidden_dim=32, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# ================== 训练函数 ==================
def train_and_evaluate(data_path, feature_dims, model_name, epochs=20, device='cpu'):
    """
    训练并评估单个模型

    返回: (准确率, F1分数, 训练历史)
    """
    print(f"\n{'='*70}")
    print(f"训练模型: {model_name} ({feature_dims}维特征)")
    print(f"{'='*70}")

    # 1. 加载数据
    dataset = ByzantineDataset(
        data_path=data_path,
        window_size=50,
        stride=50,
        feature_dims=feature_dims
    )

    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"\n数据集划分:")
    print(f"  - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. 创建模型
    model = LSTMBehaviorClassifier(input_dim=feature_dims, hidden_dim=32, num_layers=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

    # 3. 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_accuracies = []
    val_f1_scores = []

    print(f"\n开始训练...")
    start_time = time.time()

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='binary')
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    training_time = time.time() - start_time
    print(f"\n训练完成！耗时: {training_time:.1f}秒")

    # 4. 测试集评估
    print(f"\n测试集评估:")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='binary')

    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Byzantine']))

    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n混淆矩阵:")
    print(cm)

    # 5. 保存模型
    model_path = f'{model_name.replace(" ", "_")}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ 模型已保存至 {model_path}")

    return {
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores,
        'training_time': training_time,
        'model_path': model_path
    }


# ================== 对比实验主函数 ==================
def run_comparison_experiment():
    """
    运行完整的对比实验
    """
    print("="*70)
    print("对比实验：7维特征 vs 10维特征（含Correntropy）")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")

    results = {}

    # 实验1: 7维特征（原始数据）
    if os.path.exists('training_data_minimal'):
        print(f"\n{'#'*70}")
        print("实验1: 7维特征（原始方法）")
        print(f"{'#'*70}")

        results['7D'] = train_and_evaluate(
            data_path='training_data_minimal',
            feature_dims=7,
            model_name='LSTM_7D_baseline',
            epochs=20,
            device=device
        )
    else:
        print("\n⚠️  未找到 training_data_minimal，跳过7维实验")

    # 实验2: 10维特征（含Correntropy）
    if os.path.exists('training_data_correntropy'):
        print(f"\n{'#'*70}")
        print("实验2: 10维特征（含Correntropy）")
        print(f"{'#'*70}")

        results['10D'] = train_and_evaluate(
            data_path='training_data_correntropy',
            feature_dims=10,
            model_name='LSTM_10D_correntropy',
            epochs=20,
            device=device
        )
    else:
        print("\n⚠️  未找到 training_data_correntropy，跳过10维实验")
        print("   请先运行: python3 generate_correntropy_data.py")

    # ================== 对比报告 ==================
    if len(results) == 2:
        print(f"\n{'='*70}")
        print("对比报告")
        print(f"{'='*70}")

        print(f"\n{'指标':<20} {'7维特征':<15} {'10维特征':<15} {'提升':<15}")
        print(f"{'-'*70}")

        # 准确率
        acc_7d = results['7D']['test_accuracy']
        acc_10d = results['10D']['test_accuracy']
        acc_improve = (acc_10d - acc_7d) * 100
        print(f"{'测试准确率':<20} {acc_7d:<15.4f} {acc_10d:<15.4f} {acc_improve:+.2f}%")

        # F1分数
        f1_7d = results['7D']['test_f1']
        f1_10d = results['10D']['test_f1']
        f1_improve = (f1_10d - f1_7d) * 100
        print(f"{'测试F1分数':<20} {f1_7d:<15.4f} {f1_10d:<15.4f} {f1_improve:+.2f}%")

        # 训练时间
        time_7d = results['7D']['training_time']
        time_10d = results['10D']['training_time']
        time_diff = time_10d - time_7d
        print(f"{'训练时间(秒)':<20} {time_7d:<15.1f} {time_10d:<15.1f} {time_diff:+.1f}s")

        print(f"{'-'*70}")

        # ================== 可视化对比 ==================
        print(f"\n生成对比图表...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 训练损失对比
        ax = axes[0, 0]
        ax.plot(results['7D']['train_losses'], label='7D Features', linewidth=2)
        ax.plot(results['10D']['train_losses'], label='10D Features (Correntropy)', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 验证准确率对比
        ax = axes[0, 1]
        ax.plot(results['7D']['val_accuracies'], label='7D Features', linewidth=2)
        ax.plot(results['10D']['val_accuracies'], label='10D Features (Correntropy)', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Accuracy', fontsize=12)
        ax.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 验证F1分数对比
        ax = axes[1, 0]
        ax.plot(results['7D']['val_f1_scores'], label='7D Features', linewidth=2)
        ax.plot(results['10D']['val_f1_scores'], label='10D Features (Correntropy)', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation F1 Score', fontsize=12)
        ax.set_title('Validation F1 Score Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 最终指标对比
        ax = axes[1, 1]
        metrics = ['Accuracy', 'F1 Score']
        scores_7d = [acc_7d, f1_7d]
        scores_10d = [acc_10d, f1_10d]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width/2, scores_7d, width, label='7D Features', alpha=0.8)
        ax.bar(x + width/2, scores_10d, width, label='10D Features (Correntropy)', alpha=0.8)

        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Test Set Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim([0.9, 1.01])
        ax.grid(True, axis='y', alpha=0.3)

        # 添加数值标签
        for i, (v7, v10) in enumerate(zip(scores_7d, scores_10d)):
            ax.text(i - width/2, v7 + 0.002, f'{v7:.4f}', ha='center', va='bottom', fontsize=10)
            ax.text(i + width/2, v10 + 0.002, f'{v10:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('correntropy_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✓ 对比图表已保存至 correntropy_comparison.png")

    print(f"\n{'='*70}")
    print("对比实验完成！")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    results = run_comparison_experiment()
