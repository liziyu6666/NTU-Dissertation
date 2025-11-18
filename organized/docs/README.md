# Documentation Index

**Byzantine Detection in Multi-Agent Systems - Complete Documentation**

本目录包含项目的完整文档，从研究框架到数学细节，从实验说明到GitHub部署。

---

## 📚 文档分类

### 🎯 研究框架与总结

#### 1. [RESEARCH_FRAMEWORK_SUMMARY.md](RESEARCH_FRAMEWORK_SUMMARY.md) ⭐⭐⭐
**用途**: 向导师汇报、论文写作参考
**内容**:
- 完整的三层防御架构
- 所有方法的详细说明
- 实验结果总结
- 论文撰写建议
- 学术贡献分析

**适合**: 第一次阅读项目、准备汇报、撰写论文

---

#### 2. [RESEARCH_REPORT.md](RESEARCH_REPORT.md)
**用途**: 研究过程记录
**内容**:
- 研究历程
- 遇到的问题和解决方案
- 迭代改进过程

**适合**: 了解项目演化过程

---

### 📐 数学基础文档

#### 3. [BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md](BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md) ⭐⭐⭐
**用途**: 理解检测方法的数学原理
**内容**:
- **系统模型**: 多智能体动力学、通信拓扑、Byzantine攻击模型
- **RCP-f方法**: 距离过滤算法的数学推导、复杂度分析
- **ℓ1方法**: Hankel矩阵构建、凸优化求解、LP转换
- **LSTM方法**: 特征工程、Correntropy理论、网络架构
- **实际计算示例**: 每个方法的数值计算演示

**特点**:
- ✅ 完整的数学推导
- ✅ 详细的计算步骤
- ✅ Python代码示例
- ✅ 实际数值验证

**适合**:
- 想深入理解检测方法的数学原理
- 需要重现算法的研究者
- 撰写论文的Methodology部分

---

#### 4. [CONTROLLER_DESIGN_MATHEMATICS.md](CONTROLLER_DESIGN_MATHEMATICS.md) ⭐⭐
**用途**: 理解控制器设计的数学基础
**内容**:
- **合作输出调节问题**: 问题定义、标准假设
- **调节方程求解**: Sylvester方程、Kronecker积方法
- **分布式观测器**: 观测器设计、收敛性分析
- **状态反馈控制器**: 极点配置、LQR设计
- **稳定性分析**: Lyapunov方法、闭环系统分析
- **数值计算示例**: Cart-Pendulum系统的完整计算

**特点**:
- ✅ 控制理论基础
- ✅ 调节方程详细推导
- ✅ 参数选择指南
- ✅ 故障排查建议

**适合**:
- 想理解控制器如何设计
- 需要调整控制参数的研究者
- 撰写System Model和Control Design部分

---

### 🔬 方法论说明

#### 5. [CORRECT_METHOD_EXPLANATION.md](CORRECT_METHOD_EXPLANATION.md)
**用途**: 理解LSTM训练的正确方法
**内容**:
- 错误方法 vs 正确方法对比
- 为什么学习行为模式而非agent ID
- 滑动窗口数据集构建

**适合**: 初次接触LSTM Byzantine检测

---

#### 6. [CORRENTROPY_FEATURE_SUMMARY.md](CORRENTROPY_FEATURE_SUMMARY.md) ⭐
**用途**: 理解Correntropy特征增强
**内容**:
- Maximum Correntropy Criterion (MCC) 理论
- 3个Correntropy特征的物理意义
- 7维 vs 10维特征对比
- 为什么Correntropy有效

**特点**:
- ✅ 跨领域创新（联邦学习 → Byzantine检测）
- ✅ 理论+实践结合
- ✅ 详细的向导师汇报要点

**适合**:
- 理解特征工程创新
- 撰写论文的Contributions部分

---

### 📊 实验与结果

#### 7. [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)
**用途**: 实验结果总结
**内容**:
- 各场景实验结果
- 性能指标对比
- 可视化图表说明

**适合**: 撰写Experiments和Results部分

---

### 🚀 部署与使用

#### 8. [GITHUB_PUSH_GUIDE.md](GITHUB_PUSH_GUIDE.md)
**用途**: GitHub推送详细指南
**内容**:
- SSH密钥配置步骤
- Personal Access Token使用
- 故障排查方案

**适合**: 需要推送代码到GitHub

---

#### 9. [GIT_PUSH_INSTRUCTIONS.md](GIT_PUSH_INSTRUCTIONS.md)
**用途**: Git操作快速参考
**内容**:
- 常用Git命令
- 推送前检查清单

**适合**: Git命令快速查询

---

## 🗺️ 阅读路线图

### 路线1: 快速了解项目（30分钟）
```
1. RESEARCH_FRAMEWORK_SUMMARY.md (第1-3节)
   ↓ 了解项目概览和三层架构
2. QUICK_REFERENCE.md (根目录)
   ↓ 查看核心结果和关键数字
3. organized/experiments/README.md
   ↓ 了解5个实验
```

### 路线2: 深入理解方法（2-3小时）
```
1. BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md
   ↓ 逐个方法学习数学原理
2. CONTROLLER_DESIGN_MATHEMATICS.md
   ↓ 理解控制器设计
3. CORRENTROPY_FEATURE_SUMMARY.md
   ↓ 理解特征工程创新
```

### 路线3: 论文写作准备（1-2天）
```
Day 1:
  - RESEARCH_FRAMEWORK_SUMMARY.md (完整阅读)
  - BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md (Sections 2-4)
  - 整理实验结果图表

Day 2:
  - CONTROLLER_DESIGN_MATHEMATICS.md
  - RESULTS_SUMMARY.md
  - 开始撰写论文各部分
```

### 路线4: 代码复现（1周）
```
Week 1:
  Day 1-2: 理解数学基础
    - BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md
    - CONTROLLER_DESIGN_MATHEMATICS.md

  Day 3-4: 运行实验
    - organized/experiments/README.md
    - 逐个运行5个实验脚本

  Day 5-6: 训练LSTM
    - CORRECT_METHOD_EXPLANATION.md
    - 生成数据 → 训练模型 → 在线检测

  Day 7: 综合测试
    - 运行完整的6场景对比
```

---

## 📑 按主题查找

### 主题：系统模型
- **BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md** - Section 1
- **CONTROLLER_DESIGN_MATHEMATICS.md** - Section 1

### 主题：Byzantine攻击
- **BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md** - Section 1.4
- **RESEARCH_FRAMEWORK_SUMMARY.md** - System Model部分

### 主题：RCP-f算法
- **BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md** - Section 2
- **RESEARCH_FRAMEWORK_SUMMARY.md** - 方法论详解

### 主题：ℓ1优化
- **BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md** - Section 3
- **RESEARCH_FRAMEWORK_SUMMARY.md** - 方法1详解

### 主题：LSTM检测
- **BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md** - Section 4
- **CORRECT_METHOD_EXPLANATION.md**
- **CORRENTROPY_FEATURE_SUMMARY.md**

### 主题：控制器设计
- **CONTROLLER_DESIGN_MATHEMATICS.md** - Sections 2-4
- **RESEARCH_FRAMEWORK_SUMMARY.md** - System Model

### 主题：实验结果
- **RESEARCH_FRAMEWORK_SUMMARY.md** - 实验结果总结
- **RESULTS_SUMMARY.md**
- **organized/experiments/README.md**

### 主题：论文写作
- **RESEARCH_FRAMEWORK_SUMMARY.md** - Section VII
- **所有数学文档** - 作为Methodology参考

---

## 🎯 重要文档速查

### 向导师汇报前必读
1. ✅ RESEARCH_FRAMEWORK_SUMMARY.md
2. ✅ BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md (Sections 2-4)
3. ✅ organized/experiments/README.md

### 论文撰写前必读
1. ✅ RESEARCH_FRAMEWORK_SUMMARY.md (完整)
2. ✅ BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md (完整)
3. ✅ CONTROLLER_DESIGN_MATHEMATICS.md
4. ✅ RESULTS_SUMMARY.md

### 代码调试前必读
1. ✅ BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md (Section 6)
2. ✅ CONTROLLER_DESIGN_MATHEMATICS.md (Section 6)
3. ✅ organized/experiments/README.md

### GitHub推送前必读
1. ✅ GITHUB_PUSH_GUIDE.md

---

## 📊 文档统计

| 文档 | 字数 | 页数 (估) | 难度 | 重要性 |
|-----|------|----------|-----|-------|
| RESEARCH_FRAMEWORK_SUMMARY.md | ~15,000 | ~50 | ⭐⭐ | ⭐⭐⭐ |
| BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md | ~18,000 | ~60 | ⭐⭐⭐ | ⭐⭐⭐ |
| CONTROLLER_DESIGN_MATHEMATICS.md | ~12,000 | ~40 | ⭐⭐⭐ | ⭐⭐ |
| CORRENTROPY_FEATURE_SUMMARY.md | ~3,000 | ~10 | ⭐⭐ | ⭐⭐ |
| CORRECT_METHOD_EXPLANATION.md | ~2,500 | ~8 | ⭐ | ⭐⭐ |
| **总计** | **~50,000** | **~170** | - | - |

---

## 🔗 相关链接

### 项目文档
- [主README](../../README.md) - 项目主页
- [实验说明](../experiments/README.md) - 5个对比实验
- [快速参考](../../QUICK_REFERENCE.md) - 快速查询手册

### 代码目录
- [核心代码](../core/) - 仿真系统
- [实验脚本](../experiments/) - 对比实验
- [训练代码](../training/) - LSTM训练
- [检测代码](../detection/) - 在线检测

### 结果目录
- [模型文件](../results/models/) - 训练好的LSTM
- [结果图表](../results/figures/) - 可视化结果

---

## 💡 使用建议

### 第一次阅读
1. 从 **RESEARCH_FRAMEWORK_SUMMARY.md** 开始
2. 快速浏览了解整体框架
3. 根据兴趣深入特定部分

### 遇到数学问题
1. 查阅 **BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md**
2. 查看对应章节的详细推导
3. 参考Section 6的计算示例

### 遇到代码问题
1. 查阅 **organized/experiments/README.md**
2. 检查数学文档的算法伪代码
3. 对照实际代码实现

### 准备论文
1. 使用 **RESEARCH_FRAMEWORK_SUMMARY.md** 作为大纲
2. 从数学文档提取公式和推导
3. 从实验文档提取结果和图表

---

## ❓ 常见问题

**Q: 从哪个文档开始读？**
A: 如果是第一次，从 RESEARCH_FRAMEWORK_SUMMARY.md 开始；如果想深入理解，从 BYZANTINE_DETECTION_MATHEMATICAL_FOUNDATIONS.md 开始。

**Q: 数学推导看不懂怎么办？**
A: 每个方法都有"实际计算示例"章节（Section 6），从具体数字开始理解。

**Q: 如何引用这些方法？**
A: RCP-f和LSTM+Correntropy是原创方法；ℓ1方法来自Yan Jiaqi et al.的论文，需要引用。

**Q: 文档可以直接用于论文吗？**
A: 可以作为参考，但需要用学术语言重新组织。建议：Introduction用RESEARCH_FRAMEWORK, Methodology用数学文档。

---

## 📝 更新日志

- **2025-10-30**: 创建文档索引，添加数学基础文档
- **2025-10-29**: 完成研究框架总结
- **2025-10-27**: 添加Correntropy特征文档
- **2025-10-23**: 添加LSTM方法说明

---

*最后更新: 2025-10-30*
*文档总数: 9*
*总字数: ~50,000*
