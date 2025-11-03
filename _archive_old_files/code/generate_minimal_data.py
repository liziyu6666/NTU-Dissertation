"""
最小数据集生成 - 用于快速测试
只生成8个场景（每个智能体作为拜占庭节点一次）
"""

import sys
sys.path.insert(0, '/home/liziyu/d/dissertation/code')
from importlib import import_module
import pickle
import os
import time

feature_collection = import_module('1_feature_collection')
run_simulation = feature_collection.run_simulation

def generate_minimal_dataset(attack_type='sine', save_path='training_data_minimal'):
    """
    生成最小数据集：8个场景，每个智能体作为拜占庭节点

    参数:
        attack_type: 使用的攻击类型
        save_path: 保存路径
    """

    os.makedirs(save_path, exist_ok=True)

    num_agents = 8
    metadata = []

    print("="*60)
    print(f"生成最小数据集（{num_agents}个场景）")
    print(f"攻击类型: {attack_type}")
    print("="*60)

    start_time = time.time()

    for byz_node in range(num_agents):
        scenario_start = time.time()
        print(f"[{byz_node+1}/{num_agents}] Byzantine节点 = {byz_node}...", end=' ', flush=True)

        try:
            data = run_simulation(
                faulty_agent=byz_node,
                attack_type=attack_type,
                scenario_id=byz_node,
                silent=True
            )

            if data is not None:
                # 保存
                filename = f"scenario_byz{byz_node}_{attack_type}.pkl"
                filepath = os.path.join(save_path, filename)

                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)

                size_kb = os.path.getsize(filepath) / 1024
                elapsed = time.time() - scenario_start

                metadata.append({
                    'scenario_id': byz_node,
                    'faulty_agent': byz_node,
                    'attack_type': attack_type,
                    'filename': filename
                })

                print(f"✓ ({size_kb:.1f} KB, {elapsed:.1f}s)")
            else:
                print("✗ 返回None")

        except Exception as e:
            print(f"✗ 错误: {str(e)[:40]}")
            continue

    # 保存元数据
    metadata_file = os.path.join(save_path, 'metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    total_time = time.time() - start_time
    total_size = sum(os.path.getsize(os.path.join(save_path, f))
                     for f in os.listdir(save_path) if f.endswith('.pkl'))

    print("="*60)
    print(f"✓ 成功生成 {len(metadata)}/{num_agents} 个场景")
    print(f"✓ 总耗时: {total_time/60:.1f} 分钟")
    print(f"✓ 总大小: {total_size/1024/1024:.2f} MB")
    print(f"✓ 数据保存至: {save_path}/")
    print("="*60)

    return metadata


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='生成最小训练数据集')
    parser.add_argument('--attack', type=str, default='sine',
                       choices=['sine', 'constant', 'random', 'ramp', 'mixed'],
                       help='攻击类型 (默认: sine)')
    parser.add_argument('--save-path', type=str, default='training_data_minimal',
                       help='保存路径')

    args = parser.parse_args()

    print(f"\n开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    metadata = generate_minimal_dataset(
        attack_type=args.attack,
        save_path=args.save_path
    )

    print(f"\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n完成！可以使用这些数据进行训练测试。")
