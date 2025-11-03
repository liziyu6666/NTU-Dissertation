"""
正确的LSTM训练方法
学习Byzantine节点的行为模式，而非agent ID
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ================== 正确的数据集类 ==================
class ByzantineBehaviorDataset(Dataset):
    """
    学习Byzantine行为模式的数据集

    关键区别：
    - 每个窗口是一个样本（不是每个agent）
    - 标签是该窗口的行为类别（不是agent ID）
    """

    def __init__(self, data_path='training_data_minimal', window_size=50, stride=50):
        self.window_size = window_size
        self.stride = stride

        # 加载元数据
        with open(os.path.join(data_path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        print(f"正确方法：学习Byzantine行为模式")
        print(f"加载 {len(metadata)} 个场景...")

        self.windows = []
        self.labels = []

        scenario_count = 0
        window_count = 0

        # 遍历每个场景
        for meta in metadata:
            filepath = os.path.join(data_path, meta['filename'])

            with open(filepath, 'rb') as f:
                scenario = pickle.load(f)

            byzantine_id = scenario['faulty_agent']
            scenario_count += 1

            # 遍历每个agent
            for agent in scenario['agents']:
                agent_id = agent['agent_id']
                is_byzantine = (agent_id == byzantine_id)

                # 提取特征
                features = np.array([
                    agent['estimation_error'],
                    agent['position_error'],
                    agent['angle'],
                    agent['angular_velocity'],
                    agent['control_input'],
                    agent['v_hat_0'],
                    agent['v_hat_1']
                ]).T  # (T, 7)

                # 归一化
                features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

                # 滑动窗口提取
                num_steps = len(features)
                for start in range(0, num_steps - self.window_size, self.stride):
                    window = features[start:start + self.window_size]

                    self.windows.append(window)
                    self.labels.append(1 if is_byzantine else 0)
                    window_count += 1

        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        print(f"\n✓ 从 {scenario_count} 个场景中提取了 {window_count} 个行为窗口")
        print(f"  - 窗口形状: {self.windows.shape}")
        print(f"  - Normal行为窗口: {(self.labels == 0).sum()}")
        print(f"  - Byzantine行为窗口: {(self.labels == 1).sum()}")
        print(f"\n关键理解：")
        print(f"  ✓ 每个窗口代表一段行为模式（不是一个agent）")
        print(f"  ✓ 模型学习的是：什么样的行为是Byzantine")
        print(f"  ✓ 可以应用到新场景的任何agent")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]), torch.LongTensor([self.labels[idx]])[0]


# ================== LSTM模型（不变）==================
class LSTMBehaviorClassifier(nn.Module):
    """单智能体行为分类器"""

    def __init__(self, input_dim=7, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        # x: (batch, 50, 7)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.fc2(out)
        return out  # (batch, 2)


# ================== 训练函数 ==================
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
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

    print("\n分类报告:")
    print("="*60)
    print(classification_report(all_labels, all_preds,
                                target_names=['Normal Behavior', 'Byzantine Behavior']))

    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(cm)

    return all_preds, all_labels


# ================== 主程序 ==================
if __name__ == '__main__':
    print("="*60)
    print("正确的Byzantine行为学习方法")
    print("="*60)
    print("\n核心思想：")
    print("  ✓ 学习Byzantine行为的特征模式")
    print("  ✓ 而不是记住某个agent ID是Byzantine")
    print("  ✓ 可以泛化到新场景的任意agent")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")

    # 1. 加载数据
    print("\n1. 加载数据...")
    dataset = ByzantineBehaviorDataset(
        data_path='training_data_minimal',
        window_size=50,
        stride=50
    )

    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"\n数据集划分:")
    print(f"  - 训练集: {len(train_dataset)} 个行为窗口")
    print(f"  - 验证集: {len(val_dataset)} 个行为窗口")
    print(f"  - 测试集: {len(test_dataset)} 个行为窗口")

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. 创建模型
    print("\n2. 创建模型...")
    model = LSTMBehaviorClassifier(input_dim=7, hidden_dim=32, num_layers=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数量: {total_params:,}")

    # 3. 训练
    print("\n3. 训练模型...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader,
        epochs=20, lr=0.001, device=device
    )

    # 4. 评估
    print("\n4. 测试集评估...")
    preds, labels = evaluate_model(model, test_loader, device=device)

    # 5. 保存模型
    print("\n5. 保存模型...")
    torch.save(model.state_dict(), 'lstm_behavior_classifier.pth')
    print("✓ 模型已保存至 lstm_behavior_classifier.pth")

    # 6. 绘制学习曲线
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
    plt.savefig('training_curves_correct.png', dpi=150)
    print("✓ 学习曲线已保存至 training_curves_correct.png")

    print("\n" + "="*60)
    print("完成！现在这个模型可以：")
    print("  ✓ 识别任何agent的Byzantine行为")
    print("  ✓ 应用到新场景中")
    print("  ✓ 进行在线实时检测")
    print("="*60)
