"""
测试训练好的Byzantine检测器
在新的场景上测试模型性能
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import sys

sys.path.insert(0, '/home/liziyu/d/dissertation/code')
from importlib import import_module
feature_collection = import_module('1_feature_collection')
run_simulation = feature_collection.run_simulation

# 导入模型定义
from train_lstm_lite import LSTMClassifier


def prepare_features(agent_data, window_size=50):
    """准备单个智能体的特征"""

    # 提取特征
    features = np.array([
        agent_data['estimation_error'],
        agent_data['position_error'],
        agent_data['angle'],
        agent_data['angular_velocity'],
        agent_data['control_input'],
        agent_data['v_hat_0'],
        agent_data['v_hat_1']
    ]).T  # Shape: (T, 7)

    # 归一化
    features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # 提取多个窗口
    num_steps = len(features_norm)
    windows = []

    for start in range(0, num_steps - window_size, window_size):
        window = features_norm[start:start + window_size]
        windows.append(window)

    return np.array(windows, dtype=np.float32)


def detect_byzantine_nodes(scenario_data, model, device='cpu', threshold=0.5):
    """
    检测拜占庭节点

    参数:
        scenario_data: 场景数据
        model: 训练好的模型
        device: 设备
        threshold: 分类阈值

    返回:
        检测结果字典
    """

    model.eval()
    results = {}

    with torch.no_grad():
        for agent in scenario_data['agents']:
            agent_id = agent['agent_id']
            is_byzantine_true = agent['is_byzantine']

            # 准备特征
            windows = prepare_features(agent, window_size=50)

            # 预测
            windows_tensor = torch.FloatTensor(windows).to(device)
            outputs = model(windows_tensor)

            # 计算平均概率
            probs = torch.softmax(outputs, dim=1)
            avg_byzantine_prob = probs[:, 1].mean().item()

            # 判断
            is_byzantine_pred = avg_byzantine_prob > threshold

            results[agent_id] = {
                'true_label': is_byzantine_true,
                'predicted_label': is_byzantine_pred,
                'byzantine_probability': avg_byzantine_prob,
                'num_windows': len(windows)
            }

    return results


def test_on_new_scenario(byzantine_node=5, attack_type='random'):
    """测试新场景"""

    print("="*60)
    print("Byzantine节点检测器测试")
    print("="*60)

    # 加载模型
    print("\n1. 加载模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMClassifier(input_dim=7, hidden_dim=32, num_layers=1).to(device)
    model.load_state_dict(torch.load('lstm_byzantine_detector_lite.pth', map_location=device))
    print(f"✓ 模型已加载 (设备: {device})")

    # 运行新场景
    print(f"\n2. 生成测试场景...")
    print(f"   - 拜占庭节点: Agent {byzantine_node}")
    print(f"   - 攻击类型: {attack_type}")

    scenario_data = run_simulation(
        faulty_agent=byzantine_node,
        attack_type=attack_type,
        scenario_id=999,
        silent=True
    )

    if scenario_data is None:
        print("✗ 场景生成失败")
        return

    print(f"✓ 场景生成成功")

    # 检测
    print(f"\n3. 运行检测...")
    results = detect_byzantine_nodes(scenario_data, model, device=device, threshold=0.5)

    # 显示结果
    print(f"\n4. 检测结果:")
    print("="*60)
    print(f"{'Agent':<8} {'真实标签':<12} {'预测标签':<12} {'拜占庭概率':<15} {'状态'}")
    print("-"*60)

    correct = 0
    total = 0

    for agent_id in sorted(results.keys()):
        r = results[agent_id]

        true_label = "Byzantine" if r['true_label'] else "Normal"
        pred_label = "Byzantine" if r['predicted_label'] else "Normal"
        prob = r['byzantine_probability']

        status = "✓" if r['true_label'] == r['predicted_label'] else "✗"

        print(f"{agent_id:<8} {true_label:<12} {pred_label:<12} {prob:<15.4f} {status}")

        if r['true_label'] == r['predicted_label']:
            correct += 1
        total += 1

    accuracy = correct / total
    print("="*60)
    print(f"准确率: {correct}/{total} = {accuracy:.2%}")

    # 总结
    detected_nodes = [aid for aid, r in results.items() if r['predicted_label']]
    print(f"\n检测到的拜占庭节点: {detected_nodes}")
    print(f"真实的拜占庭节点: [{byzantine_node}]")

    if byzantine_node in detected_nodes:
        print("✓ 成功检测到拜占庭节点！")
    else:
        print("✗ 未能检测到拜占庭节点")

    print("="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='测试Byzantine检测器')
    parser.add_argument('--byzantine', type=int, default=5,
                       help='拜占庭节点ID (0-7)')
    parser.add_argument('--attack', type=str, default='random',
                       choices=['sine', 'constant', 'random', 'ramp', 'mixed'],
                       help='攻击类型')

    args = parser.parse_args()

    test_on_new_scenario(
        byzantine_node=args.byzantine,
        attack_type=args.attack
    )
