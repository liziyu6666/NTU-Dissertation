"""
使用LSTM检测拜占庭节点
包含两种方案：
1. 异常检测（LSTM Autoencoder）
2. 分类（LSTM Classifier）
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from scipy.integrate import solve_ivp
import sys

# 导入原始仿真代码的参数
sys.path.append('/home/liziyu/d/dissertation/code')


# ==================== 数据生成 ====================
class SimulationDataGenerator:
    """生成多组仿真数据用于训练"""

    def __init__(self, num_agents=8, f=1):
        self.num_agents = num_agents
        self.f = f

        # 物理参数
        self.m = [0.1 * (i + 1) for i in range(num_agents)]
        self.M = [1.0 * (i + 1) for i in range(num_agents)]
        self.l = [0.1 * (i + 1) for i in range(num_agents)]
        self.g = 9.8
        self.friction = 0.15
        self.S = np.array([[0, 1], [-1, 0]])

        # 通信拓扑
        self.adj_matrix = np.zeros((num_agents, num_agents), dtype=int)
        self.adj_matrix[0:4, 0:4] = 1
        np.fill_diagonal(self.adj_matrix[0:4, 0:4], 0)
        self.adj_matrix[4:, 0:4] = 1
        self.adj_matrix[4, 5] = self.adj_matrix[5, 6] = self.adj_matrix[6, 7] = 1

        self.agents = None
        self._initialize_agents()

    def _initialize_agents(self):
        """初始化智能体（简化版，复用原代码逻辑）"""
        from collections import namedtuple
        Agent = namedtuple('Agent', ['A', 'B', 'E', 'C', 'F', 'K11', 'K12', 'Xi', 'Ui'])

        self.agents = []
        for i in range(self.num_agents):
            # 简化：使用固定的增益矩阵
            mi, Mi, li = self.m[i], self.M[i], self.l[i]

            A = np.array([
                [0, 1, 0, 0],
                [0, 0, self.g, 0],
                [0, 0, 0, 1],
                [0, self.friction/(li*Mi), (Mi+mi)*self.g/(li*Mi), -self.friction/Mi]
            ])
            B = np.array([[0], [0], [0], [1.0/(li*Mi)]])
            E = np.array([[0, 0], [2.0/Mi, 0], [0, 0], [1.0/(li*Mi), 0]])
            C = np.array([[1, 0, -li, 0]])
            F = np.array([[-1, 0]])

            # 简化的增益（这里使用固定值）
            K11 = np.array([-100-20*i, -50-10*i, -200-40*i, -50-10*i])
            K12 = np.array([[0.5, 1.0]])
            Xi = np.eye(4, 2) * 0.1
            Ui = np.array([[0.1, 0.2]])

            self.agents.append(Agent(A, B, E, C, F, K11, K12, Xi, Ui))

    def generate_scenario(self, faulty_agent, attack_type='mixed', duration=15, noise_level=0.0):
        """
        生成一个仿真场景

        参数:
            faulty_agent: 拜占庭节点编号
            attack_type: 攻击类型 ('mixed', 'sine', 'ramp', 'constant', 'random')
            duration: 仿真时长
            noise_level: 噪声水平

        返回:
            data: dict包含所有节点的时序数据
        """

        def byzantine_attack(t, attack_type):
            """不同类型的拜占庭攻击"""
            if attack_type == 'mixed':
                return np.array([50 * np.sin(10 * t) + 15 * np.cos(12 * t), t / 15])
            elif attack_type == 'sine':
                return np.array([30 * np.sin(8 * t), 30 * np.cos(8 * t)])
            elif attack_type == 'ramp':
                return np.array([t * 2, -t * 1.5])
            elif attack_type == 'constant':
                return np.array([100, -50])
            elif attack_type == 'random':
                return np.random.randn(2) * 20
            else:
                return np.array([0, 0])

        def apply_rcpf_filter(v_hat_i, neighbor_vhats, f):
            """RCP-f滤波器"""
            if len(neighbor_vhats) == 0:
                return np.array([]).reshape(0, 2)

            neighbor_vhats = np.array(neighbor_vhats)
            n_neighbors = len(neighbor_vhats)

            if n_neighbors <= 2 * f:
                return neighbor_vhats

            distances = np.linalg.norm(neighbor_vhats - v_hat_i, axis=1)
            sorted_indices = np.argsort(distances)
            keep_indices = sorted_indices[:n_neighbors - f]

            return neighbor_vhats[keep_indices]

        def total_system(t, y):
            """系统动力学"""
            states = y.reshape(self.num_agents, 6)
            dvdt = np.zeros((self.num_agents, 6))
            v_real = np.array([np.cos(t), np.sin(t)])

            for i in range(self.num_agents):
                x = states[i, :4]
                v_hat = states[i, 4:6]
                neighbors = np.where(self.adj_matrix[i] == 1)[0]
                is_target_node = (i < 4)

                if i == faulty_agent:
                    dv_hat = byzantine_attack(t, attack_type)
                else:
                    neighbor_vhats = [states[j, 4:6] for j in neighbors]
                    filtered_neighbors = apply_rcpf_filter(v_hat, neighbor_vhats, self.f)

                    gain = 50.0
                    consensus_term = np.zeros(2)
                    if len(filtered_neighbors) > 0:
                        filtered_mean = np.mean(filtered_neighbors, axis=0)
                        consensus_term = gain * (filtered_mean - v_hat)

                    if is_target_node:
                        consensus_term += gain * 2.0 * (v_real - v_hat)

                    dv_hat = self.S @ v_hat + consensus_term

                # 简化的动力学
                agent = self.agents[i]
                u = agent.K11 @ x + agent.K12.flatten() @ v_hat
                dxdt = (agent.A @ x + agent.B.flatten() * u + agent.E @ v_hat).flatten()

                dvdt[i, :4] = dxdt
                dvdt[i, 4:6] = dv_hat

            return dvdt.flatten()

        # 初始条件
        y0 = np.zeros(self.num_agents * 6)
        for i in range(self.num_agents):
            y0[i * 6] = 0.1 + 0.01 * i + np.random.randn() * noise_level
            y0[i * 6 + 2] = 0.05 + np.random.randn() * noise_level * 0.1
            y0[i * 6 + 4:i * 6 + 6] = [1.0, 0.0]

        # 仿真
        t_span = (0, duration)
        t_eval = np.linspace(*t_span, int(duration * 50))  # 50 Hz采样

        sol = solve_ivp(total_system, t_span, y0, t_eval=t_eval, method='RK45',
                       rtol=1e-6, atol=1e-8, max_step=0.02)

        # 提取特征
        data = {
            't': sol.t,
            'agents': []
        }

        v_real = np.vstack((np.cos(sol.t), np.sin(sol.t)))

        for i in range(self.num_agents):
            agent_data = {
                'id': i,
                'is_byzantine': (i == faulty_agent),
                'position': sol.y[i * 6],  # x1 = h + l*θ
                'velocity': sol.y[i * 6 + 1],  # x2
                'angle': sol.y[i * 6 + 2],  # θ
                'angular_vel': sol.y[i * 6 + 3],  # dθ/dt
                'v_hat': sol.y[i * 6 + 4:i * 6 + 6],  # 估计值 [v01, v02]
                'position_error': np.abs(sol.y[i * 6] - np.cos(sol.t)),
                'estimation_error': np.linalg.norm(sol.y[i * 6 + 4:i * 6 + 6] - v_real, axis=0)
            }
            data['agents'].append(agent_data)

        return data


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
        # x: (batch, seq_len, input_dim)

        # 编码
        _, (hidden, cell) = self.encoder(x)

        # 使用编码器的最终隐藏状态初始化解码器
        # 重复hidden状态以匹配序列长度
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
        self.fc2 = nn.Linear(32, 2)  # 二分类：正常/拜占庭

    def forward(self, x):
        # x: (batch, seq_len, input_dim)

        # LSTM处理序列
        lstm_out, (hidden, _) = self.lstm(x)

        # 使用最后时刻的隐藏状态
        last_hidden = hidden[-1]  # (batch, hidden_dim)

        # 全连接层
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# ==================== 数据集定义 ====================

class TimeSeriesDataset(Dataset):
    """时序数据集"""

    def __init__(self, sequences, labels=None):
        """
        sequences: list of (seq_len, feature_dim) arrays
        labels: list of binary labels (optional, for classification)
        """
        self.sequences = [torch.FloatTensor(seq) for seq in sequences]
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        else:
            return self.sequences[idx]


def extract_features(agent_data, window_size=100, stride=50):
    """
    从单个智能体数据中提取滑动窗口特征

    返回:
        windows: list of (window_size, feature_dim) arrays
    """
    # 选择特征：位置误差、估计误差、v_hat的两个分量、角度、角速度
    features = np.vstack([
        agent_data['position_error'],
        agent_data['estimation_error'],
        agent_data['v_hat'][0],
        agent_data['v_hat'][1],
        agent_data['angle'],
        agent_data['angular_vel']
    ]).T  # (time_steps, 6)

    # 归一化
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # 滑动窗口
    windows = []
    for i in range(0, len(features) - window_size, stride):
        windows.append(features[i:i+window_size])

    return windows


# ==================== 训练函数 ====================

def train_autoencoder(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """训练LSTM自编码器"""

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    train_losses = []
    val_losses = []

    print("开始训练LSTM自编码器...")

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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses


def train_classifier(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """训练LSTM分类器"""

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("开始训练LSTM分类器...")

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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


# ==================== 主程序 ====================

if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ==================== 步骤1：生成训练数据 ====================
    print("\n" + "="*60)
    print("步骤1：生成仿真数据")
    print("="*60)

    generator = SimulationDataGenerator(num_agents=8, f=1)

    # 生成多个场景（不同的拜占庭节点和攻击类型）
    num_scenarios = 20  # 每个配置生成20个场景
    attack_types = ['mixed', 'sine', 'ramp', 'constant']

    all_data = []

    for faulty_agent in range(8):  # 每个节点都可能是拜占庭节点
        for attack_type in attack_types:
            for _ in range(num_scenarios // len(attack_types)):
                print(f"生成场景: 拜占庭节点={faulty_agent}, 攻击类型={attack_type}")
                data = generator.generate_scenario(
                    faulty_agent=faulty_agent,
                    attack_type=attack_type,
                    duration=15,
                    noise_level=0.01
                )
                all_data.append(data)

    print(f"✓ 共生成 {len(all_data)} 个场景")

    # ==================== 步骤2：提取特征 ====================
    print("\n" + "="*60)
    print("步骤2：提取时序特征")
    print("="*60)

    window_size = 100  # 2秒窗口 (50 Hz采样)
    stride = 25  # 0.5秒滑动

    # 方案A: 自编码器（只用正常节点数据）
    normal_sequences = []

    # 方案B: 分类器（需要标签）
    all_sequences = []
    all_labels = []

    for data in all_data:
        for agent_data in data['agents']:
            windows = extract_features(agent_data, window_size, stride)

            for window in windows:
                all_sequences.append(window)
                all_labels.append(1 if agent_data['is_byzantine'] else 0)

                if not agent_data['is_byzantine']:
                    normal_sequences.append(window)

    print(f"✓ 提取特征完成")
    print(f"  - 正常节点窗口数: {len(normal_sequences)}")
    print(f"  - 总窗口数: {len(all_sequences)}")
    print(f"  - 拜占庭窗口数: {sum(all_labels)}")
    print(f"  - 特征维度: {normal_sequences[0].shape}")

    # 划分训练集和验证集
    split_ratio = 0.8

    # 自编码器数据
    split_idx_ae = int(len(normal_sequences) * split_ratio)
    train_sequences_ae = normal_sequences[:split_idx_ae]
    val_sequences_ae = normal_sequences[split_idx_ae:]

    # 分类器数据
    indices = np.random.permutation(len(all_sequences))
    split_idx_clf = int(len(all_sequences) * split_ratio)
    train_idx = indices[:split_idx_clf]
    val_idx = indices[split_idx_clf:]

    train_sequences_clf = [all_sequences[i] for i in train_idx]
    train_labels_clf = [all_labels[i] for i in train_idx]
    val_sequences_clf = [all_sequences[i] for i in val_idx]
    val_labels_clf = [all_labels[i] for i in val_idx]

    # 创建数据加载器
    train_dataset_ae = TimeSeriesDataset(train_sequences_ae)
    val_dataset_ae = TimeSeriesDataset(val_sequences_ae)
    train_loader_ae = DataLoader(train_dataset_ae, batch_size=32, shuffle=True)
    val_loader_ae = DataLoader(val_dataset_ae, batch_size=32)

    train_dataset_clf = TimeSeriesDataset(train_sequences_clf, train_labels_clf)
    val_dataset_clf = TimeSeriesDataset(val_sequences_clf, val_labels_clf)
    train_loader_clf = DataLoader(train_dataset_clf, batch_size=32, shuffle=True)
    val_loader_clf = DataLoader(val_dataset_clf, batch_size=32)

    # ==================== 步骤3：训练模型 ====================
    print("\n" + "="*60)
    print("步骤3：训练模型")
    print("="*60)

    input_dim = 6  # 6个特征

    # 方案A: 训练自编码器
    print("\n【方案A：异常检测 - LSTM自编码器】")
    model_ae = LSTMAutoencoder(input_dim=input_dim, hidden_dim=64, num_layers=2)
    train_losses_ae, val_losses_ae = train_autoencoder(
        model_ae, train_loader_ae, val_loader_ae,
        epochs=50, lr=0.001, device=device
    )

    # 方案B: 训练分类器
    print("\n【方案B：分类 - LSTM分类器】")
    model_clf = LSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2)
    train_losses_clf, val_losses_clf, train_accs_clf, val_accs_clf = train_classifier(
        model_clf, train_loader_clf, val_loader_clf,
        epochs=50, lr=0.001, device=device
    )

    # ==================== 步骤4：评估和可视化 ====================
    print("\n" + "="*60)
    print("步骤4：评估模型性能")
    print("="*60)

    # 评估自编码器
    print("\n【方案A：异常检测评估】")
    model_ae.eval()
    reconstruction_errors = []
    true_labels_ae = []

    with torch.no_grad():
        for i, seq in enumerate(all_sequences):
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
            reconstructed = model_ae(seq_tensor)
            error = torch.mean((reconstructed - seq_tensor) ** 2).item()
            reconstruction_errors.append(error)
            true_labels_ae.append(all_labels[i])

    reconstruction_errors = np.array(reconstruction_errors)
    true_labels_ae = np.array(true_labels_ae)

    # 使用阈值检测异常
    threshold = np.percentile(reconstruction_errors[true_labels_ae == 0], 95)  # 正常数据的95分位数
    predicted_ae = (reconstruction_errors > threshold).astype(int)

    print(f"检测阈值: {threshold:.6f}")
    print(f"\n分类报告:")
    print(classification_report(true_labels_ae, predicted_ae,
                                target_names=['正常', '拜占庭']))

    # 评估分类器
    print("\n【方案B：分类评估】")
    model_clf.eval()
    all_predictions = []
    all_true_labels = []
    all_probs = []

    with torch.no_grad():
        for sequences, labels in val_loader_clf:
            sequences = sequences.to(device)
            outputs = model_clf(sequences)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 拜占庭节点的概率

    print(f"\n分类报告:")
    print(classification_report(all_true_labels, all_predictions,
                                target_names=['正常', '拜占庭']))

    # ==================== 步骤5：可视化结果 ====================
    print("\n" + "="*60)
    print("步骤5：可视化结果")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. 训练损失曲线 - 自编码器
    axes[0, 0].plot(train_losses_ae, label='Train')
    axes[0, 0].plot(val_losses_ae, label='Validation')
    axes[0, 0].set_title('LSTM Autoencoder - Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. 重构误差分布 - 自编码器
    axes[0, 1].hist(reconstruction_errors[true_labels_ae == 0], bins=50, alpha=0.7, label='Normal', density=True)
    axes[0, 1].hist(reconstruction_errors[true_labels_ae == 1], bins=50, alpha=0.7, label='Byzantine', density=True)
    axes[0, 1].axvline(threshold, color='r', linestyle='--', label=f'Threshold={threshold:.4f}')
    axes[0, 1].set_title('Reconstruction Error Distribution')
    axes[0, 1].set_xlabel('Reconstruction Error')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale('log')

    # 3. 混淆矩阵 - 自编码器
    cm_ae = confusion_matrix(true_labels_ae, predicted_ae)
    sns.heatmap(cm_ae, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2],
                xticklabels=['Normal', 'Byzantine'],
                yticklabels=['Normal', 'Byzantine'])
    axes[0, 2].set_title('Autoencoder - Confusion Matrix')
    axes[0, 2].set_ylabel('True Label')
    axes[0, 2].set_xlabel('Predicted Label')

    # 4. 训练损失曲线 - 分类器
    axes[1, 0].plot(train_losses_clf, label='Train Loss')
    axes[1, 0].plot(val_losses_clf, label='Val Loss')
    axes[1, 0].set_title('LSTM Classifier - Training Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Cross Entropy Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 5. 准确率曲线 - 分类器
    axes[1, 1].plot(train_accs_clf, label='Train Accuracy')
    axes[1, 1].plot(val_accs_clf, label='Val Accuracy')
    axes[1, 1].set_title('LSTM Classifier - Training Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # 6. 混淆矩阵 - 分类器
    cm_clf = confusion_matrix(all_true_labels, all_predictions)
    sns.heatmap(cm_clf, annot=True, fmt='d', cmap='Greens', ax=axes[1, 2],
                xticklabels=['Normal', 'Byzantine'],
                yticklabels=['Normal', 'Byzantine'])
    axes[1, 2].set_title('Classifier - Confusion Matrix')
    axes[1, 2].set_ylabel('True Label')
    axes[1, 2].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('byzantine_detection_lstm_results.png', dpi=150, bbox_inches='tight')
    print("✓ 结果图已保存至 byzantine_detection_lstm_results.png")

    # 保存模型
    torch.save(model_ae.state_dict(), 'lstm_autoencoder.pth')
    torch.save(model_clf.state_dict(), 'lstm_classifier.pth')
    print("✓ 模型已保存")

    print("\n" + "="*60)
    print("训练和评估完成！")
    print("="*60)
