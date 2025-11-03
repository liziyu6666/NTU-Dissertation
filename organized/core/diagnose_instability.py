"""
诊断脚本：分析系统在第15秒失稳的原因
"""
import numpy as np
import matplotlib.pyplot as plt

# 分析拜占庭攻击信号
t = np.linspace(0, 15, 750)

# 原始攻击信号（dv_hat，即v_hat的导数）
byzantine_dv_hat_1 = 50 * np.sin(10 * t) + 15 * np.cos(12 * t)
byzantine_dv_hat_2 = t / 15  # 这个会导致二次增长！

# 积分得到 v_hat（假设初始值为[1.0, 0.0]）
dt = t[1] - t[0]
v_hat_1_integrated = np.cumsum(byzantine_dv_hat_1) * dt + 1.0
v_hat_2_integrated = np.cumsum(byzantine_dv_hat_2) * dt + 0.0

# 真实参考信号
v_real_1 = np.cos(t)
v_real_2 = np.sin(t)

# 创建诊断图
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('拜占庭攻击信号分析：为什么系统在15秒失稳？', fontsize=14, fontweight='bold')

# 1. dv_hat[0] (速度导数的第一分量)
axes[0, 0].plot(t, byzantine_dv_hat_1, 'r-', linewidth=2)
axes[0, 0].set_title('拜占庭节点：dv_hat[0] = 50*sin(10t) + 15*cos(12t)')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('dv_hat[0]')
axes[0, 0].grid(True)
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# 2. dv_hat[1] (速度导数的第二分量) - 问题所在！
axes[0, 1].plot(t, byzantine_dv_hat_2, 'r-', linewidth=2)
axes[0, 1].set_title('拜占庭节点：dv_hat[1] = t/15 (线性增长！)')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('dv_hat[1]')
axes[0, 1].grid(True)
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].text(7.5, 0.5, '⚠️ 线性增长的导数\n→ 二次增长的v_hat!',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                fontsize=10, ha='center')

# 3. 积分后的 v_hat[0]
axes[1, 0].plot(t, v_hat_1_integrated, 'r-', linewidth=2, label='拜占庭 v_hat[0]')
axes[1, 0].plot(t, v_real_1, 'k--', linewidth=2, label='真实 v[0] = cos(t)')
axes[1, 0].set_title('积分后：v_hat[0] vs 真实值')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('v_hat[0]')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. 积分后的 v_hat[1] - 二次增长！
axes[1, 1].plot(t, v_hat_2_integrated, 'r-', linewidth=2, label='拜占庭 v_hat[1]')
axes[1, 1].plot(t, v_real_2, 'k--', linewidth=2, label='真实 v[1] = sin(t)')
axes[1, 1].set_title('积分后：v_hat[1] vs 真实值 (二次增长！)')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('v_hat[1]')
axes[1, 1].legend()
axes[1, 1].grid(True)
axes[1, 1].text(7.5, 0.3, f'⚠️ t=15时: {v_hat_2_integrated[-1]:.2f}\n(应该是-1到1之间！)',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                fontsize=9, ha='center', color='white')

# 5. 估计误差范数
estimation_error = np.sqrt((v_hat_1_integrated - v_real_1)**2 +
                          (v_hat_2_integrated - v_real_2)**2)
axes[2, 0].plot(t, estimation_error, 'r-', linewidth=2)
axes[2, 0].set_title('估计误差: ||v_hat - v_real||')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('Error Norm')
axes[2, 0].grid(True)
axes[2, 0].set_yscale('log')
axes[2, 0].text(7.5, 1e1, f'⚠️ t=15时: {estimation_error[-1]:.2e}',
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8),
                fontsize=10, ha='center')

# 6. 控制输入的影响（假设 K12 = [k1, k2]，E = [0, pi1, 0, pi2]^T）
# 假设 K12 的典型值
K12_example = np.array([10.0, 5.0])
pi1_example = 2.0  # 典型值
pi2_example = 1.0

# 控制项: K12 @ v_hat
control_contribution = K12_example[0] * v_hat_1_integrated + K12_example[1] * v_hat_2_integrated

# 外部信号项: E @ v_hat 对第二个状态的影响
external_contribution = pi1_example * v_hat_1_integrated

axes[2, 1].plot(t, control_contribution, 'b-', linewidth=2, label='控制项: K12 @ v_hat')
axes[2, 1].plot(t, external_contribution, 'g-', linewidth=2, label='外部项: pi1 * v_hat[0]')
axes[2, 1].set_title('v_hat 对系统的影响')
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].set_ylabel('Contribution')
axes[2, 1].legend()
axes[2, 1].grid(True)
axes[2, 1].text(7.5, 40, '⚠️ 随着v_hat增长\n控制输入失控！',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                fontsize=10, ha='center', color='white')

plt.tight_layout()
plt.savefig('byzantine_attack_diagnosis.png', dpi=150, bbox_inches='tight')
print("✅ 诊断图已保存到 byzantine_attack_diagnosis.png")

# 打印数值分析
print("\n" + "="*60)
print("数值分析报告")
print("="*60)
print(f"\n1. 拜占庭攻击信号分析 (t=15秒):")
print(f"   - dv_hat[0] = {byzantine_dv_hat_1[-1]:.2f} (有界振荡)")
print(f"   - dv_hat[1] = {byzantine_dv_hat_2[-1]:.2f} (线性增长到1.0)")
print(f"\n2. 积分后的 v_hat 值 (t=15秒):")
print(f"   - v_hat[0] ≈ {v_hat_1_integrated[-1]:.2f} (应该在-1到1之间)")
print(f"   - v_hat[1] ≈ {v_hat_2_integrated[-1]:.2f} (应该在-1到1之间)")
print(f"   - 真实值: v[0]={v_real_1[-1]:.2f}, v[1]={v_real_2[-1]:.2f}")
print(f"\n3. 估计误差:")
print(f"   - ||v_hat - v_real|| = {estimation_error[-1]:.2e}")
print(f"\n4. 问题根源:")
print(f"   ❌ dv_hat[1] = t/15 是线性函数")
print(f"   ❌ 积分后 v_hat[1] ~ t²/30 (二次增长)")
print(f"   ❌ 控制器使用 v_hat: u = K11@x + K12@v_hat")
print(f"   ❌ 当 v_hat 线性增长时，控制输入也线性增长")
print(f"   ❌ 系统动力学: dx/dt = Ax + Bu + Ev_hat")
print(f"   ❌ E@v_hat 项会导致状态爆炸性增长")
print(f"\n5. 为什么在15秒失稳？")
print(f"   - 初期：RCP-f 过滤器能够抵抗拜占庭攻击")
print(f"   - 中期：v_hat 累积误差逐渐增大")
print(f"   - 后期：当 v_hat 偏差超过某个阈值，过滤器失效")
print(f"   - 临界点：~12-15秒，v_hat 增长到 {v_hat_2_integrated[-1]:.2f}")
print(f"   - 失稳：控制输入和外部项同时爆炸")
print(f"\n6. 解决方案:")
print(f"   ✅ 方案1: 修改拜占庭攻击为有界信号")
print(f"      dv_hat = [50*sin(10t) + 15*cos(12t), 5*sin(3t)]")
print(f"   ✅ 方案2: 减小观测器增益 (当前gain=50太大)")
print(f"   ✅ 方案3: 增加 v_hat 饱和限制")
print(f"   ✅ 方案4: 改进 RCP-f 过滤器的容忍度")
print(f"   ✅ 方案5: 减少仿真时间到 10 秒")
print("="*60)

# 额外分析：在什么时刻系统开始不稳定？
print(f"\n额外分析：系统稳定性边界")
print("-"*60)

# 寻找 v_hat 偏差超过阈值的时刻
threshold_values = [2, 5, 10, 20, 50]
for thresh in threshold_values:
    idx = np.where(estimation_error > thresh)[0]
    if len(idx) > 0:
        t_cross = t[idx[0]]
        print(f"误差超过 {thresh:>3}: t = {t_cross:>6.3f} 秒")

print("-"*60)
