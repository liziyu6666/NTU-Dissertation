"""
验证修复：分析修复后的拜占庭攻击信号
"""
import numpy as np
import matplotlib.pyplot as plt

# 时间轴
t = np.linspace(0, 15, 750)

# 修复后的攻击信号（dv_hat）
byzantine_dv_hat_1_fixed = 50 * np.sin(10 * t) + 15 * np.cos(12 * t)
byzantine_dv_hat_2_fixed = 5 * np.sin(3 * t)  # 修复后：有界振荡

# 修复前的攻击信号（对比）
byzantine_dv_hat_2_old = t / 15  # 旧版本：线性增长

# 积分得到 v_hat
dt = t[1] - t[0]
v_hat_1_fixed = np.cumsum(byzantine_dv_hat_1_fixed) * dt + 1.0
v_hat_2_fixed = np.cumsum(byzantine_dv_hat_2_fixed) * dt + 0.0
v_hat_2_old = np.cumsum(byzantine_dv_hat_2_old) * dt + 0.0

# 真实参考信号
v_real_1 = np.cos(t)
v_real_2 = np.sin(t)

# 创建对比图
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Verification: Fixed Byzantine Attack Signal', fontsize=14, fontweight='bold')

# 1. dv_hat[0] - 没有变化
axes[0, 0].plot(t, byzantine_dv_hat_1_fixed, 'r-', linewidth=2)
axes[0, 0].set_title('dv_hat[0] = 50*sin(10t) + 15*cos(12t)\n(Unchanged)')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('dv_hat[0]')
axes[0, 0].grid(True)
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# 2. dv_hat[1] - 修复前 vs 修复后
axes[0, 1].plot(t, byzantine_dv_hat_2_old, 'r--', linewidth=2, label='OLD: t/15 (Linear!)', alpha=0.7)
axes[0, 1].plot(t, byzantine_dv_hat_2_fixed, 'g-', linewidth=2, label='FIXED: 5*sin(3t) (Bounded!)')
axes[0, 1].set_title('dv_hat[1]: Before vs After Fix')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('dv_hat[1]')
axes[0, 1].legend()
axes[0, 1].grid(True)
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# 3. v_hat[0] 积分后
axes[0, 2].plot(t, v_hat_1_fixed, 'r-', linewidth=2, label='Byzantine v_hat[0]')
axes[0, 2].plot(t, v_real_1, 'k--', linewidth=2, label='True v[0] = cos(t)')
axes[0, 2].set_title('Integrated v_hat[0] vs True Value')
axes[0, 2].set_xlabel('Time (s)')
axes[0, 2].set_ylabel('v_hat[0]')
axes[0, 2].legend()
axes[0, 2].grid(True)

# 4. v_hat[1] 积分后 - 关键对比
axes[1, 0].plot(t, v_hat_2_old, 'r--', linewidth=2, label=f'OLD: {v_hat_2_old[-1]:.2f} @ t=15', alpha=0.7)
axes[1, 0].plot(t, v_hat_2_fixed, 'g-', linewidth=2, label=f'FIXED: {v_hat_2_fixed[-1]:.2f} @ t=15')
axes[1, 0].plot(t, v_real_2, 'k--', linewidth=1.5, label='True v[1] = sin(t)')
axes[1, 0].set_title('Integrated v_hat[1]: Before vs After Fix')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('v_hat[1]')
axes[1, 0].legend()
axes[1, 0].grid(True)
axes[1, 0].text(7.5, 5, f'✅ Fixed: stays bounded!\nOLD: {v_hat_2_old[-1]:.2f}\nFIXED: {v_hat_2_fixed[-1]:.2f}',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontsize=9, ha='center')

# 5. 估计误差对比
error_old = np.sqrt((v_hat_1_fixed - v_real_1)**2 + (v_hat_2_old - v_real_2)**2)
error_fixed = np.sqrt((v_hat_1_fixed - v_real_1)**2 + (v_hat_2_fixed - v_real_2)**2)

axes[1, 1].plot(t, error_old, 'r--', linewidth=2, label=f'OLD: {error_old[-1]:.2e} @ t=15', alpha=0.7)
axes[1, 1].plot(t, error_fixed, 'g-', linewidth=2, label=f'FIXED: {error_fixed[-1]:.2e} @ t=15')
axes[1, 1].set_title('Estimation Error: ||v_hat - v_real||')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Error Norm')
axes[1, 1].legend()
axes[1, 1].grid(True)
axes[1, 1].set_yscale('log')

# 6. 控制输入影响对比
K12_example = np.array([10.0, 5.0])
control_old = K12_example[0] * v_hat_1_fixed + K12_example[1] * v_hat_2_old
control_fixed = K12_example[0] * v_hat_1_fixed + K12_example[1] * v_hat_2_fixed

axes[1, 2].plot(t, control_old, 'r--', linewidth=2, label=f'OLD: {control_old[-1]:.1f} @ t=15', alpha=0.7)
axes[1, 2].plot(t, control_fixed, 'g-', linewidth=2, label=f'FIXED: {control_fixed[-1]:.1f} @ t=15')
axes[1, 2].set_title('Control Contribution: K12 @ v_hat')
axes[1, 2].set_xlabel('Time (s)')
axes[1, 2].set_ylabel('Control Input')
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout()
plt.savefig('fix_verification.png', dpi=150, bbox_inches='tight')
print("✅ Verification plot saved to fix_verification.png")

# 打印对比报告
print("\n" + "="*70)
print("VERIFICATION REPORT: Before vs After Fix")
print("="*70)

print(f"\n1. Attack Signal dv_hat[1] @ t=15:")
print(f"   OLD:   {byzantine_dv_hat_2_old[-1]:.4f} (linear growth)")
print(f"   FIXED: {byzantine_dv_hat_2_fixed[-1]:.4f} (bounded oscillation)")

print(f"\n2. Integrated v_hat[1] @ t=15:")
print(f"   OLD:   {v_hat_2_old[-1]:.4f} (quadratic growth → UNSTABLE)")
print(f"   FIXED: {v_hat_2_fixed[-1]:.4f} (bounded → STABLE)")
print(f"   TRUE:  {v_real_2[-1]:.4f}")

print(f"\n3. Estimation Error @ t=15:")
print(f"   OLD:   {error_old[-1]:.4e}")
print(f"   FIXED: {error_fixed[-1]:.4e}")
print(f"   Reduction: {(error_old[-1] - error_fixed[-1]) / error_old[-1] * 100:.1f}%")

print(f"\n4. Control Input Contribution @ t=15:")
print(f"   OLD:   {control_old[-1]:.2f}")
print(f"   FIXED: {control_fixed[-1]:.2f}")
print(f"   Reduction: {abs(control_old[-1] - control_fixed[-1]):.2f}")

print(f"\n5. Signal Properties:")
print(f"   dv_hat[1] OLD:")
print(f"      - Type: Linear (t/15)")
print(f"      - Min: {byzantine_dv_hat_2_old.min():.4f}")
print(f"      - Max: {byzantine_dv_hat_2_old.max():.4f}")
print(f"      - Mean: {byzantine_dv_hat_2_old.mean():.4f}")
print(f"\n   dv_hat[1] FIXED:")
print(f"      - Type: Sinusoidal (5*sin(3t))")
print(f"      - Min: {byzantine_dv_hat_2_fixed.min():.4f}")
print(f"      - Max: {byzantine_dv_hat_2_fixed.max():.4f}")
print(f"      - Mean: {byzantine_dv_hat_2_fixed.mean():.4f}")

print(f"\n6. Expected System Behavior:")
print(f"   ✅ Fixed signal is bounded: |dv_hat[1]| ≤ 5")
print(f"   ✅ Integrated v_hat stays bounded: |v_hat[1]| ≤ 10")
print(f"   ✅ Control input should remain stable")
print(f"   ✅ System should NOT diverge at t=15")

print(f"\n7. Why the simulation MIGHT still show divergence:")
print(f"   - Check if simulation file was re-run after edit")
print(f"   - Check if correct file is being executed")
print(f"   - Check if there are cached .pyc files")
print(f"   - Check for other numerical instabilities")

print("="*70)
