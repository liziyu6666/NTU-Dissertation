"""
内存友好的批量训练数据生成
逐个保存场景，避免内存溢出
"""

import numpy as np
import pickle
import sys
import os
import gc

# 导入仿真代码
sys.path.insert(0, '/home/liziyu/d/dissertation/code')
from importlib import import_module
feature_collection = import_module('1_feature_collection')
run_simulation = feature_collection.run_simulation

def generate_scenarios_lite(num_scenarios_per_config=1, save_path='training_data'):
    """
    内存友好版本：逐个生成并保存场景

    参数:
        num_scenarios_per_config: 每个配置生成的场景数（建议1-2，避免内存问题）
        save_path: 保存路径
    """

    # 减少攻击类型，先测试
    attack_types = ['sine', 'constant', 'random']  # 从5种减少到3种
    num_agents = 8

    os.makedirs(save_path, exist_ok=True)

    print("="*60)
    print("生成训练数据集（内存友好版）")
    print("="*60)
    print(f"配置:")
    print(f"  - 智能体数量: {num_agents}")
    print(f"  - 攻击类型: {attack_types}")
    print(f"  - 每个配置的场景数: {num_scenarios_per_config}")
    print(f"  - 总场景数: {num_agents * len(attack_types) * num_scenarios_per_config}")
    print(f"  - 保存路径: {save_path}/")
    print()

    total = num_agents * len(attack_types) * num_scenarios_per_config
    completed = 0
    success_count = 0

    scenario_metadata = []  # 只保存元数据，不保存完整数据

    for faulty_agent in range(num_agents):
        for attack_type in attack_types:
            for rep in range(num_scenarios_per_config):
                scenario_id = faulty_agent * len(attack_types) * num_scenarios_per_config + \
                              attack_types.index(attack_type) * num_scenarios_per_config + rep

                try:
                    # 运行仿真
                    print(f"[{completed+1}/{total}] Byzantine={faulty_agent}, Attack={attack_type}...", end=' ')

                    data = run_simulation(
                        faulty_agent=faulty_agent,
                        attack_type=attack_type,
                        scenario_id=scenario_id,
                        silent=True
                    )

                    if data is not None:
                        # 立即保存到单独文件
                        filename = f"scenario_{scenario_id:03d}_byz{faulty_agent}_{attack_type}.pkl"
                        filepath = os.path.join(save_path, filename)

                        with open(filepath, 'wb') as f:
                            pickle.dump(data, f)

                        # 记录元数据
                        scenario_metadata.append({
                            'scenario_id': scenario_id,
                            'faulty_agent': faulty_agent,
                            'attack_type': attack_type,
                            'filename': filename
                        })

                        success_count += 1
                        file_size = os.path.getsize(filepath) / 1024  # KB
                        print(f"✓ 已保存 ({file_size:.1f} KB)")

                        # 清理内存
                        del data
                        gc.collect()

                    completed += 1

                except Exception as e:
                    print(f"✗ 错误: {str(e)[:50]}")
                    completed += 1
                    continue

    # 保存元数据索引
    metadata_file = os.path.join(save_path, 'metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(scenario_metadata, f)

    print(f"\n{'='*60}")
    print(f"✓ 成功生成 {success_count}/{total} 个场景")
    print(f"✓ 元数据已保存至 {metadata_file}")

    # 统计
    total_size = sum(os.path.getsize(os.path.join(save_path, f))
                     for f in os.listdir(save_path) if f.endswith('.pkl'))
    print(f"✓ 总数据大小: {total_size / 1024 / 1024:.2f} MB")
    print("="*60)

    return scenario_metadata


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='生成训练数据（内存友好版）')
    parser.add_argument('--num-per-config', type=int, default=1,
                       help='每个配置生成的场景数 (默认: 1, 建议<=2)')
    parser.add_argument('--save-path', type=str, default='training_data',
                       help='保存路径 (默认: training_data)')

    args = parser.parse_args()

    if args.num_per_config > 2:
        print(f"警告: num_per_config={args.num_per_config} 可能导致运行时间过长")
        print("建议使用 1 或 2")

    metadata = generate_scenarios_lite(
        num_scenarios_per_config=args.num_per_config,
        save_path=args.save_path
    )

    print("\n完成！")