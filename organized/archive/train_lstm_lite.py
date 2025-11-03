"""
轻量级LSTM训练脚本
使用最小数据集，避免内存溢出
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ================== 数据集类 ==================
class ByzantineDataset(Dataset):
    """时间序列数据集"""

    def __init__(self, data_path='training_data_minimal', window_size=50, stride=25):
        """
        参数:
            data_path: 数据路径
            window_size: 时间窗口大小（降低以减少内存）
            stride: 滑动步长
        """
        self.window_size = window_size
        self.stride = stride

        # 加载元数据
        with open(os.path.join(data_path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        print(f"加载 {len(metadata)} 个场景...")

        self.windows = []
        self.labels = []

        # 逐个加载场景，提取窗口
        for meta in metadata:
            filepath = os.path.join(data_path, meta['filename'])

            with open(filepath, 'rb') as f:
                scenario = pickle.load(f)

            # 处理每个智能体
            for agent in scenario['agents']:
                agent_id = agent['agent_id']
                is_byzantine = agent['is_byzantine']

                # 提取特征（排除元数据）
                features = np.array([
                    agent['estimation_error'],
                    agent['position_error'],
                    agent['angle'],
                    agent['angular_velocity'],
                    agent['control_input'],
                    agent['v_hat_0'],
                    agent['v_hat_1']
                ]).T  # Shape: (T, 7)

                # 归一化
                features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

                # 提取窗口（步长采样以减少数据量）
                num_steps = len(features)
                for start in range(0, num_steps - self.window_size, self.stride):
                    window = features[start:start + self.window_size]

                    self.windows.append(window)
                    self.labels.append(1 if is_byzantine else 0)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        print(f"✓ 提取 {len(self.windows)} 个窗口")
        print(f"  - 特征维度: {self.windows.shape}")
        print(f"  - 正常样本: {(self.labels == 0).sum()}")
        print(f"  - 拜占庭样本: {(self.labels == 1).sum()}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]), torch.LongTensor([self.labels[idx]])[0]


# ================== LSTM分类器 ==================
class LSTMClassifier(nn.Module):
    """简化的LSTM分类器"""

    def __init__(self, input_dim=7, hidden_dim=32, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=0 if num_layers == 1 else 0.2)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 使用最后一个隐藏状态
        last_hidden = h_n[-1]  # (batch, hidden_dim)

        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.fc2(out)

        return out


# ================== 训练函数 ==================
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
    """训练模型"""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_accuracies = []

    print("\n开始训练...")
    print("="*60)

    for epoch in range(epochs):
        # 训练
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

        # 验证
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)

                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_acc = correct / total
        val_accuracies.append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

    print("="*60)
    print("训练完成！")

    return train_losses, val_accuracies


# ================== 评估函数 ==================
def evaluate_model(model, test_loader, device='cpu'):
    """评估模型"""

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

    # 分类报告
    print("\n分类报告:")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Byzantine']))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(cm)

    return all_preds, all_labels


# ================== 主程序 ==================
if __name__ == '__main__':
    print("LSTM Byzantine节点检测器 - 轻量版")
    print(f"开始时间: {time.strftime('%H:%M:%S')}")
    print("="*60)

    # 检查CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 加载数据（减少window_size和增大stride以减少内存）
    print("\n1. 加载数据...")
    dataset = ByzantineDataset(
        data_path='training_data_minimal',
        window_size=50,   # 从100降到50
        stride=50         # 增大步长减少窗口数量
    )

    # 划分训练/验证/测试集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"\n数据集划分:")
    print(f"  - 训练集: {len(train_dataset)}")
    print(f"  - 验证集: {len(val_dataset)}")
    print(f"  - 测试集: {len(test_dataset)}")

    # 创建DataLoader（减少batch_size）
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 创建模型（减小hidden_dim）
    print("\n2. 创建模型...")
    model = LSTMClassifier(input_dim=7, hidden_dim=32, num_layers=1).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数量: {total_params:,}")

    # 训练模型
    print("\n3. 训练模型...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader,
        epochs=20,
        lr=0.001,
        device=device
    )

    # 评估模型
    print("\n4. 测试集评估...")
    preds, labels = evaluate_model(model, test_loader, device=device)

    # 保存模型
    print("\n5. 保存模型...")
    torch.save(model.state_dict(), 'lstm_byzantine_detector_lite.pth')
    print("✓ 模型已保存至 lstm_byzantine_detector_lite.pth")

    # 绘制学习曲线
    print("\n6. 生成学习曲线...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)

    ax2.plot(val_accuracies)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves_lite.png', dpi=150)
    print("✓ 学习曲线已保存至 training_curves_lite.png")

    print("\n" + "="*60)
    print(f"结束时间: {time.strftime('%H:%M:%S')}")
    print("完成！")
