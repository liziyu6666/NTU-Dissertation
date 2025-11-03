"""
生成包含Correntropy特征的训练数据
"""

import sys
sys.path.insert(0, '/home/liziyu/d/dissertation/code')
from importlib import import_module
import pickle
import os
import time
import gc

# 导入新的特征收集模块
feature_collection_corr = import_module('1_feature_collection_correntropy')
run_simulation = feature_collection_corr.run_simulation

def generate_correntropy_dataset(num_scenarios_per_config=1, attack_type='sine', save_path='training_data_correntropy'):
    """
    生成包含Correntropy特征的数据集

    特征对比:
        原始: 7维 [estimation_error, position_error, angle, angular_velocity,
                  control_input, v_hat_0, v_hat_1]
        新增: 10维 [上述7维 + avg_correntropy, min_correntropy, std_correntropy]
    """

    os.makedirs(save_path, exist_ok=True)

    num_agents = 8
    metadata = []

    print("="*70)
    print("生成Correntropy特征数据集")
    print("="*70)
    print(f"配置:")
    print(f"  - 智能体数量: {num_agents}")
    print(f"  - 攻击类型: {attack_type}")
    print(f"  - 每个Byzantine节点的场景数: {num_scenarios_per_config}")
    print(f"  - 总场景数: {num_agents * num_scenarios_per_config}")
    print(f"  - 特征维度: 10 (7个原始 + 3个Correntropy)")
    print(f"  - 保存路径: {save_path}/")
    print("="*70)

    start_time = time.time()
    success_count = 0

    for rep in range(num_scenarios_per_config):
        for byzantine_node in range(num_agents):
            scenario_id = rep * num_agents + byzantine_node

            print(f"\n[{scenario_id+1}/{num_agents * num_scenarios_per_config}] ", end='')
            print(f"Byzantine={byzantine_node}, Rep={rep+1}...", end=' ', flush=True)

            try:
                scenario_start = time.time()

                data = run_simulation(
                    faulty_agent=byzantine_node,
                    attack_type=attack_type,
                    scenario_id=scenario_id,
                    silent=True
                )

                if data is not None:
                    # 保存到单独文件
                    filename = f"scenario_{scenario_id:03d}_byz{byzantine_node}_{attack_type}_corr.pkl"
                    filepath = os.path.join(save_path, filename)

                    with open(filepath, 'wb') as f:
                        pickle.dump(data, f)

                    # 记录元数据
                    metadata.append({
                        'scenario_id': scenario_id,
                        'faulty_agent': byzantine_node,
                        'attack_type': attack_type,
                        'filename': filename
                    })

                    size_kb = os.path.getsize(filepath) / 1024
                    elapsed = time.time() - scenario_start
                    success_count += 1

                    print(f"✓ ({size_kb:.1f} KB, {elapsed:.1f}s)")

                    # 清理内存
                    del data
                    gc.collect()
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

    print(f"\n{'='*70}")
    print(f"✓ 成功生成 {success_count}/{num_agents * num_scenarios_per_config} 个场景")
    print(f"✓ 总耗时: {total_time/60:.1f} 分钟")
    print(f"✓ 总大小: {total_size/1024/1024:.2f} MB")
    print(f"✓ 元数据保存至: {metadata_file}")
    print("="*70)

    return metadata


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='生成Correntropy特征数据')
    parser.add_argument('--num-per-config', type=int, default=1,
                       help='每个配置生成的场景数 (默认: 1)')
    parser.add_argument('--attack', type=str, default='sine',
                       choices=['sine', 'constant', 'random', 'ramp', 'mixed'],
                       help='攻击类型')
    parser.add_argument('--save-path', type=str, default='training_data_correntropy',
                       help='保存路径')

    args = parser.parse_args()

    print(f"\n开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    metadata = generate_correntropy_dataset(
        num_scenarios_per_config=args.num_per_config,
        attack_type=args.attack,
        save_path=args.save_path
    )

    print(f"\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n完成！Correntropy特征数据已准备好用于训练。")
