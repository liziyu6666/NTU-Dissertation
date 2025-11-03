"""
在线Byzantine检测演示
展示如何在系统运行过程中实时检测Byzantine节点
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/liziyu/d/dissertation/code')
from importlib import import_module

# 导入模型定义
from train_lstm_correct import LSTMBehaviorClassifier

# ================== 在线检测器 ==================
class OnlineByzantineDetector:
    """
    在线Byzantine检测器

    用法：
    1. 初始化：detector = OnlineByzantineDetector(model_path)
    2. 每个时间步：detector.update(t, agents_features)
    3. 检测：result = detector.detect()  # 返回每个agent的Byzantine概率
    """

    def __init__(self, model_path, num_agents=8, window_size=50):
        """
        Args:
            model_path: 训练好的模型路径
            num_agents: 智能体数量
            window_size: 时间窗口大小（必须与训练时一致）
        """
        self.model = LSTMBehaviorClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        self.num_agents = num_agents
        self.window_size = window_size

        # 为每个agent维护一个特征缓冲区
        self.buffers = {i: [] for i in range(num_agents)}

        # 统计信息
        self.num_features = 7

    def update(self, t, agents_features):
        """
        更新每个agent的特征缓冲区

        Args:
            t: 当前时间（仅用于记录）
            agents_features: dict[int, dict]
                {
                    agent_id: {
                        'estimation_error': float,
                        'position_error': float,
                        'angle': float,
                        'angular_velocity': float,
                        'control_input': float,
                        'v_hat_0': float,
                        'v_hat_1': float
                    }
                }
        """
        for agent_id, features in agents_features.items():
            # 提取7维特征向量
            feature_vec = np.array([
                features['estimation_error'],
                features['position_error'],
                features['angle'],
                features['angular_velocity'],
                features['control_input'],
                features['v_hat_0'],
                features['v_hat_1']
            ])

            self.buffers[agent_id].append(feature_vec)

            # 保持窗口大小
            if len(self.buffers[agent_id]) > self.window_size:
                self.buffers[agent_id].pop(0)

    def detect(self):
        """
        执行Byzantine检测

        Returns:
            dict 或 None:
            如果数据不足（<window_size步），返回None
            否则返回:
            {
                agent_id: {
                    'is_byzantine': bool,      # 是否判定为Byzantine
                    'confidence': float,       # Byzantine的置信度 [0, 1]
                    'buffer_size': int         # 当前缓冲区大小
                }
            }
        """
        # 检查数据是否充足
        if len(self.buffers[0]) < self.window_size:
            return None

        results = {}

        with torch.no_grad():
            for agent_id in range(self.num_agents):
                # 获取窗口数据
                window = np.array(self.buffers[agent_id])  # (50, 7)

                # 归一化（与训练时保持一致）
                mean = window.mean(axis=0)
                std = window.std(axis=0) + 1e-8
                window_norm = (window - mean) / std

                # 转换为tensor
                window_tensor = torch.FloatTensor(window_norm).unsqueeze(0)  # (1, 50, 7)

                # 预测
                output = self.model(window_tensor)  # (1, 2)
                probs = torch.softmax(output, dim=1)[0]  # (2,)

                normal_prob = probs[0].item()
                byzantine_prob = probs[1].item()

                results[agent_id] = {
                    'is_byzantine': byzantine_prob > 0.5,
                    'confidence': byzantine_prob,
                    'buffer_size': len(self.buffers[agent_id])
                }

        return results

    def reset(self):
        """重置所有缓冲区"""
        self.buffers = {i: [] for i in range(self.num_agents)}


# ================== 演示函数 ==================
def run_online_detection_demo(byzantine_node=5, attack_type='random'):
    """
    运行在线检测演示

    Args:
        byzantine_node: 指定哪个节点是Byzantine
        attack_type: 攻击类型
    """
    print("="*70)
    print("在线Byzantine检测演示")
    print("="*70)

    # 1. 加载训练好的模型
    print("\n[步骤1] 加载训练好的检测器...")
    try:
        detector = OnlineByzantineDetector(
            model_path='lstm_behavior_classifier.pth',
            num_agents=8,
            window_size=50
        )
        print("✓ 检测器加载成功")
    except FileNotFoundError:
        print("✗ 模型文件未找到，请先运行 train_lstm_correct.py")
        return

    # 2. 生成新场景
    print(f"\n[步骤2] 生成新测试场景...")
    print(f"  - Byzantine节点: Agent {byzantine_node}")
    print(f"  - 攻击类型: {attack_type}")

    feature_collection = import_module('1_feature_collection')

    scenario = feature_collection.run_simulation(
        faulty_agent=byzantine_node,
        attack_type=attack_type,
        scenario_id=999,
        silent=True
    )

    if scenario is None:
        print("✗ 场景生成失败")
        return

    print(f"✓ 场景生成成功 (时间步数: {len(scenario['agents'][0]['time'])})")

    # 3. 逐步feed数据并实时检测
    print(f"\n[步骤3] 开始在线检测...")
    print(f"  提示：需要积累{detector.window_size}步数据后才能开始检测")

    num_steps = len(scenario['agents'][0]['time'])

    # 记录检测历史
    detection_timeline = []

    for step in range(num_steps):
        # 提取当前时间步的所有agent特征
        agents_features = {}
        for agent_id in range(8):
            agent = scenario['agents'][agent_id]
            agents_features[agent_id] = {
                'estimation_error': agent['estimation_error'][step],
                'position_error': agent['position_error'][step],
                'angle': agent['angle'][step],
                'angular_velocity': agent['angular_velocity'][step],
                'control_input': agent['control_input'][step],
                'v_hat_0': agent['v_hat_0'][step],
                'v_hat_1': agent['v_hat_1'][step]
            }

        # 更新检测器
        detector.update(scenario['agents'][0]['time'][step], agents_features)

        # 尝试检测
        result = detector.detect()

        if result is not None:
            # 记录结果
            detection_timeline.append({
                'step': step,
                'time': scenario['agents'][0]['time'][step],
                'detected': [aid for aid, r in result.items() if r['is_byzantine']],
                'confidences': {aid: r['confidence'] for aid, r in result.items()}
            })

            # 每2000步打印一次
            if step % 2000 == 0 or step == num_steps - 1:
                print(f"\n  时间步 {step}/{num_steps} (t={scenario['agents'][0]['time'][step]:.2f}s):")

                detected = [aid for aid, r in result.items() if r['is_byzantine']]
                print(f"    检测到Byzantine: {detected}")

                print(f"    各agent置信度:")
                for aid in range(8):
                    conf = result[aid]['confidence']
                    marker = "🔴" if result[aid]['is_byzantine'] else "🟢"
                    true_marker = " ← 真实Byzantine" if aid == byzantine_node else ""
                    print(f"      {marker} Agent {aid}: {conf:.3f}{true_marker}")

    # 4. 最终结果分析
    print(f"\n{'='*70}")
    print("[步骤4] 最终检测结果")
    print(f"{'='*70}")

    final_result = detector.detect()

    if final_result is None:
        print("✗ 数据不足，无法检测")
        return

    detected_final = [aid for aid, r in final_result.items() if r['is_byzantine']]

    print(f"\n检测结果:")
    print(f"  - 检测到的Byzantine节点: {detected_final}")
    print(f"  - 真实的Byzantine节点: [{byzantine_node}]")

    # 准确性评估
    true_positive = byzantine_node in detected_final
    false_positives = [aid for aid in detected_final if aid != byzantine_node]

    print(f"\n性能评估:")
    if true_positive:
        print(f"  ✓ 成功识别Byzantine节点 (Agent {byzantine_node})")
    else:
        print(f"  ✗ 未能识别Byzantine节点 (Agent {byzantine_node})")

    if len(false_positives) > 0:
        print(f"  ⚠ 误报: {false_positives}")
    else:
        print(f"  ✓ 无误报")

    # 5. 可视化检测过程
    print(f"\n[步骤5] 生成检测过程可视化...")

    if len(detection_timeline) > 0:
        # 提取时间序列
        steps = [d['step'] for d in detection_timeline]
        times = [d['time'] for d in detection_timeline]

        # 每个agent的置信度时间序列
        fig, ax = plt.subplots(figsize=(12, 6))

        for agent_id in range(8):
            confidences = [d['confidences'][agent_id] for d in detection_timeline]

            linestyle = '--' if agent_id == byzantine_node else '-'
            linewidth = 2.5 if agent_id == byzantine_node else 1.0
            label = f"Agent {agent_id} (Byzantine)" if agent_id == byzantine_node else f"Agent {agent_id}"

            ax.plot(times, confidences, label=label, linestyle=linestyle, linewidth=linewidth)

        ax.axhline(y=0.5, color='r', linestyle=':', label='检测阈值')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Byzantine Confidence', fontsize=12)
        ax.set_title(f'Online Byzantine Detection Timeline (Byzantine: Agent {byzantine_node})', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('online_detection_timeline.png', dpi=150)
        print("✓ 检测过程图已保存至 online_detection_timeline.png")

    print(f"\n{'='*70}")
    print("演示完成！")
    print(f"{'='*70}")

    print("\n关键观察:")
    print("  1. 检测器在积累50步数据后立即开始工作")
    print("  2. Byzantine节点的置信度持续偏高")
    print("  3. 可以在系统运行过程中实时识别恶意节点")
    print("  4. 不需要等到仿真结束，也不需要知道Byzantine是哪个")

    return detection_timeline


# ================== 主程序 ==================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='在线Byzantine检测演示')
    parser.add_argument('--byzantine', type=int, default=5,
                       help='Byzantine节点ID (0-7)')
    parser.add_argument('--attack', type=str, default='random',
                       choices=['sine', 'constant', 'random', 'ramp', 'mixed'],
                       help='攻击类型')

    args = parser.parse_args()

    # 运行演示
    timeline = run_online_detection_demo(
        byzantine_node=args.byzantine,
        attack_type=args.attack
    )

    if timeline is not None:
        print(f"\n总结：在 {len(timeline)} 个检测点中跟踪了Byzantine行为")
