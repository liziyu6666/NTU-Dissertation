"""
批量生成训练数据
生成50+个场景，涵盖不同的拜占庭节点和攻击类型
"""

import numpy as np
import pickle
import sys
import os

# 导入仿真代码
sys.path.insert(0, '/home/liziyu/d/dissertation/code')
from importlib import import_module
feature_collection = import_module('1_feature_collection')
run_simulation = feature_collection.run_simulation

def generate_all_scenarios(num_scenarios_per_config=2, save_path='training_data'):
    """
    生成所有训练场景

    参数:
        num_scenarios_per_config: 每个配置（拜占庭节点+攻击类型）生成的场景数
        save_path: 保存路径
    """

    attack_types = ['mixed', 'sine', 'ramp', 'constant', 'random']
    num_agents = 8

    all_scenarios = []
    scenario_id = 0

    print("="*60)
    print("生成训练数据集")
    print("="*60)
    print(f"配置:")
    print(f"  - 智能体数量: {num_agents}")
    print(f"  - 攻击类型: {attack_types}")
    print(f"  - 每个配置的场景数: {num_scenarios_per_config}")
    print(f"  - 总场景数: {num_agents * len(attack_types) * num_scenarios_per_config}")
    print()

    # 进度条
    total = num_agents * len(attack_types) * num_scenarios_per_config
    completed = 0

    for faulty_agent in range(num_agents):
        for attack_type in attack_types:
            for rep in range(num_scenarios_per_config):
                try:
                    # 运行仿真
                    data = run_simulation(
                        faulty_agent=faulty_agent,
                        attack_type=attack_type,
                        scenario_id=scenario_id,
                        silent=True
                    )

                    if data is not None:
                        all_scenarios.append(data)
                        scenario_id += 1

                    completed += 1
                    if completed % 5 == 0 or completed == total:
                        print(f"  进度: {completed}/{total} ({100*completed/total:.1f}%) - "
                              f"Byzantine={faulty_agent}, Attack={attack_type}, 成功={len(all_scenarios)}")

                except Exception as e:
                    print(f"  错误: Byzantine={faulty_agent}, Attack={attack_type}, Error={str(e)[:30]}")
                    completed += 1
                    continue

    print(f"\n✓ 成功生成 {len(all_scenarios)} 个场景")

    # 统计信息
    byzantine_counts = {}
    attack_counts = {}

    for scenario in all_scenarios:
        byz = scenario['faulty_agent']
        att = scenario['attack_type']

        byzantine_counts[byz] = byzantine_counts.get(byz, 0) + 1
        attack_counts[att] = attack_counts.get(att, 0) + 1

    print(f"\n统计信息:")
    print(f"拜占庭节点分布:")
    for agent_id in sorted(byzantine_counts.keys()):
        print(f"  Agent {agent_id}: {byzantine_counts[agent_id]} 场景")

    print(f"\n攻击类型分布:")
    for attack in sorted(attack_counts.keys()):
        print(f"  {attack}: {attack_counts[attack]} 场景")

    # 保存数据
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, 'all_scenarios.pkl')

    with open(output_file, 'wb') as f:
        pickle.dump(all_scenarios, f)

    print(f"\n✓ 数据已保存至 {output_file}")
    print(f"  文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

    return all_scenarios


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='生成训练数据')
    parser.add_argument('--num-per-config', type=int, default=2,
                       help='每个配置生成的场景数 (默认: 2)')
    parser.add_argument('--save-path', type=str, default='training_data',
                       help='保存路径 (默认: training_data)')

    args = parser.parse_args()

    scenarios = generate_all_scenarios(
        num_scenarios_per_config=args.num_per_config,
        save_path=args.save_path
    )

    print("\n" + "="*60)
    print("完成！")
    print("="*60)
