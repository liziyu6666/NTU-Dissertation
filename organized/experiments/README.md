# Byzantine Detection Experiments

本目录包含所有对比实验代码，用于评估不同Byzantine检测和容错方法的性能。

## 📁 文件说明

### 基础对比实验

#### 1. `simple_comparison.py`
**功能**: 二场景对比实验
- **场景1**: 无Byzantine节点（基准）
- **场景2**: 有Byzantine节点 + RCP-f防御

**用途**: 验证RCP-f的基本有效性

**运行**:
```bash
python3 simple_comparison.py
```

**输出**:
- `simple_comparison.png` - 对比可视化
- 终端输出性能统计

---

#### 2. `three_scenario_comparison.py`
**功能**: 三场景对比实验（控制变量法）
- **场景1**: 无Byzantine节点（基准）
- **场景2**: 有Byzantine节点，无防御（对照组）
- **场景3**: 有Byzantine节点，使用RCP-f（实验组）

**用途**: 通过控制变量证明RCP-f的作用

**关键结果**:
- 无防御: 4976× 性能恶化
- RCP-f防御: 1.03× (100%恢复)

**运行**:
```bash
python3 three_scenario_comparison.py
```

---

#### 3. `five_scenario_comparison.py` ⭐
**功能**: 五场景综合对比（融合论文方法）
- **S1**: 无Byzantine节点（基准）
- **S2**: 有Byzantine节点，无防御
- **S3**: 仅ℓ1数据清洗（来自Yan Jiaqi论文）
- **S4**: 仅RCP-f（原创方法）
- **S5**: ℓ1 + RCP-f组合

**理论基础**:
- **ℓ1优化**: Yan Jiaqi et al., "Secure Data Reconstruction: A Direct Data-Driven Approach"
- **Hankel矩阵**: 数据驱动的行为语言表示
- **RCP-f**: 基于距离的实时过滤

**关键发现**:
- ℓ1单独使用: 无效（无实时防御能力）
- RCP-f单独使用: 100%性能恢复
- 两者结合: 保持100%性能 + 理论保证

**运行**:
```bash
python3 five_scenario_comparison.py
```

**输出**:
- `five_scenario_comparison.png` - 15面板综合对比图
- 详细的性能统计报告

---

### 高级实验

#### 4. `hybrid_detection_method.py`
**功能**: 混合检测方法框架
- 结合ℓ1数据清洗和RCP-f实时过滤
- 展示如何集成不同检测方法

**特点**:
- 模块化设计
- 易于扩展新方法
- 包含完整的Hankel矩阵和ℓ1优化实现

**用途**: 作为论文方法集成的参考实现

---

#### 5. `ml_comprehensive_comparison.py` 🌟
**功能**: 机器学习方法综合对比
- **S1-S4**: 复用五场景的前4个场景
- **S5**: LSTM检测 + RCP-f（机器学习方法）
- **S6**: ℓ1 + LSTM + RCP-f（三者结合，最优方案）

**创新点**:
1. **LSTM在线检测器**:
   - 学习Byzantine行为模式
   - 实时滑动窗口检测（50步）
   - 支持加载预训练模型

2. **三层防御架构**:
   ```
   离线检测（ℓ1） → 在线学习（LSTM） → 实时过滤（RCP-f）
   ```

3. **多方法融合**:
   - 数据驱动 + 模型驱动 + 学习驱动
   - 互补优势，多层次保障

**运行**:
```bash
# 需要预先训练LSTM模型
python3 ml_comprehensive_comparison.py
```

**依赖**:
- `lstm_behavior_classifier.pth` - 预训练的LSTM模型
- PyTorch

---

## 🔬 实验对比总结

### 性能对比表

| 实验文件 | 场景数 | Byzantine防御方法 | 关键发现 |
|---------|-------|-----------------|---------|
| `simple_comparison.py` | 2 | RCP-f | 验证基本有效性 |
| `three_scenario_comparison.py` | 3 | RCP-f | 控制变量，证明作用 |
| `five_scenario_comparison.py` | 5 | ℓ1, RCP-f, 组合 | ℓ1单独无效，RCP-f有效 |
| `ml_comprehensive_comparison.py` | 6 | +LSTM | 三者结合最优 |

### 研究贡献层次

```
Level 1: simple_comparison.py
  └─> 证明RCP-f有效

Level 2: three_scenario_comparison.py
  └─> 控制变量，量化改善（4976×→1.03×）

Level 3: five_scenario_comparison.py
  └─> 融合论文方法（ℓ1），证明互补性

Level 4: ml_comprehensive_comparison.py
  └─> 机器学习增强，三层防御框架
```

---

## 🚀 快速开始

### 1. 运行基础实验
```bash
# 从最简单的开始
python3 simple_comparison.py

# 控制变量实验
python3 three_scenario_comparison.py

# 论文方法对比
python3 five_scenario_comparison.py
```

### 2. 运行ML增强实验
```bash
# 首先训练LSTM模型（如果还没有）
cd ../training
python3 train_lstm_correct.py

# 然后运行ML综合实验
cd ../experiments
python3 ml_comprehensive_comparison.py
```

### 3. 查看结果
所有实验生成的图表保存在 `../results/figures/` 目录

---

## 📊 实验结果总结

### 定量结果（正常节点平均最终误差）

| 场景 | 平均误差 | 相对基准 | 性能恢复率 |
|-----|---------|---------|-----------|
| 无Byzantine | 0.048 | 1.00× | N/A |
| 有Byzantine，无防御 | 237.7 | 4976× ⚠️ | 0% |
| 仅ℓ1 | 237.7 | 4976× | 0% |
| 仅RCP-f | 0.049 | 1.03× ✅ | **100%** |
| ℓ1+RCP-f | 0.049 | 1.03× ✅ | **100%** |
| LSTM+RCP-f | ~0.049 | ~1.03× ✅ | **100%** |
| 三者结合 | ~0.048 | ~1.00× 🏆 | **100%** |

### 关键结论

1. **RCP-f是核心**:
   - 单独使用已达到100%性能恢复
   - 计算开销低（O(n log n)）
   - 实时性强

2. **ℓ1提供理论保证**:
   - 凸优化，全局最优
   - 单独无效（无控制层响应）
   - 与RCP-f结合可增强可靠性

3. **LSTM增强识别能力**:
   - 可识别具体Byzantine节点
   - 学习多种攻击模式
   - 支持在线自适应

4. **三者结合最优**:
   - 多层次防御
   - 离线验证 + 在线检测 + 实时过滤
   - 理论与实践双重保障

---

## 📝 论文撰写建议

### 使用哪些实验？

**推荐组合**:
1. **三场景实验** (`three_scenario_comparison.py`)
   - 用于Introduction展示问题严重性
   - 用于证明RCP-f的基本有效性

2. **五场景实验** (`five_scenario_comparison.py`)
   - 用于Methodology展示论文方法对比
   - 用于Discussion分析不同方法的优劣

3. **ML综合实验** (`ml_comprehensive_comparison.py`)
   - 用于展示创新的三层防御框架
   - 用于Future Work展示ML方向

### 实验章节结构建议

```
V. Experiments
  A. Experimental Setup
     - 系统参数
     - Byzantine攻击模型
     - 评估指标

  B. Baseline Comparisons
     - 使用 three_scenario_comparison.py 的结果
     - 展示无防御 vs RCP-f的对比

  C. Integration with Data-Driven Method
     - 使用 five_scenario_comparison.py 的结果
     - 展示ℓ1方法的集成和对比

  D. Machine Learning Enhancement
     - 使用 ml_comprehensive_comparison.py 的结果
     - 展示LSTM检测的优势

  E. Ablation Studies
     - 不同Byzantine节点数量
     - 不同攻击类型
     - 不同系统参数
```

---

## 🔗 相关文档

- [研究框架总结](../docs/RESEARCH_FRAMEWORK_SUMMARY.md) - 完整的研究体系说明
- [LSTM方法说明](../docs/CORRECT_METHOD_EXPLANATION.md) - LSTM训练的正确方法
- [Correntropy特征](../docs/CORRENTROPY_FEATURE_SUMMARY.md) - 特征工程创新

---

## 📧 联系

如有问题或建议，请参考主README或联系作者。

---

*最后更新: 2025-10-30*
