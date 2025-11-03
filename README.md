# Byzantine Fault-Tolerant Multi-Agent System

**NTU Dissertation Project**: Multi-layer Byzantine Detection and Resilient Consensus Control

**Author**: liziyu
**Repository**: https://github.com/liziyu6666/NTU-Dissertation

---

## 🎯 Project Overview

This research develops a **multi-layer Byzantine fault tolerance framework** for cooperative multi-agent systems, integrating:

1. **Data-Driven Detection** (ℓ1 optimization from Yan Jiaqi's paper)
2. **Real-Time Filtering** (RCP-f algorithm - original contribution)
3. **Machine Learning Detection** (LSTM with Correntropy features - original contribution)

### Key Results
- ✅ **100% performance recovery** (4976× degradation → 1.03× baseline)
- ✅ **99% LSTM detection accuracy**
- ✅ **Real-time online detection** (<1 second latency)

---

## 📁 Repository Structure

```
dissertation/
├── organized/                    # 👈 Main codebase (START HERE)
│   ├── experiments/              # All comparison experiments
│   ├── core/                     # Simulation system
│   ├── training/                 # LSTM model training
│   ├── detection/                # Online detection
│   ├── docs/                     # Complete documentation
│   └── results/                  # Models and figures
│
├── RESEARCH_FRAMEWORK_SUMMARY.md # 📖 Complete research overview
├── README.md                     # This file
└── 论文/                         # Paper and references
```

---

## 🚀 Quick Start

### Option 1: Run Pre-built Experiments

Navigate to the experiments directory and run comparison tests:

```bash
cd organized/experiments

# Basic 2-scenario comparison
python3 simple_comparison.py

# 3-scenario with control variable
python3 three_scenario_comparison.py

# 5-scenario with ℓ1 method from paper
python3 five_scenario_comparison.py

# 6-scenario with LSTM integration
python3 ml_comprehensive_comparison.py
```

### Option 2: Train LSTM Model from Scratch

```bash
cd organized

# Step 1: Generate training data
cd data_generation
python3 generate_correntropy_data.py --attack sine

# Step 2: Train LSTM model
cd ../training
python3 train_lstm_correct.py

# Step 3: Run online detection demo
cd ../detection
python3 online_detection_demo.py
```

---

## 📊 Experimental Results Summary

### Six-Scenario Comparison

| Scenario | Description | Defense Method | Avg Error | Recovery |
|----------|-------------|----------------|-----------|----------|
| S1 | No Byzantine (baseline) | N/A | 0.048 | N/A |
| S2 | Byzantine, no defense | None | 237.7 | 0% ⚠️ |
| S3 | Byzantine + ℓ1 only | Data-driven | 237.7 | 0% |
| S4 | Byzantine + RCP-f only | Real-time filter | 0.049 | **100%** ✅ |
| S5 | Byzantine + LSTM+RCP-f | ML + filter | ~0.049 | **100%** ✅ |
| S6 | Byzantine + All three | Multi-layer | ~0.048 | **100%** 🏆 |

**Key Findings**:
- ℓ1 alone: Ineffective for real-time defense
- RCP-f alone: Achieves 100% performance recovery
- LSTM+RCP-f: Same performance + Byzantine node identification
- Combined approach: Multi-layer protection with theoretical guarantees

---

## 🔬 Research Contributions

### 1. Three-Layer Defense Architecture

```
┌─────────────────────────────────┐
│   Layer 1: Data-Driven (ℓ1)    │
│   - Hankel matrix construction  │
│   - Convex optimization         │
│   - Offline validation          │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   Layer 2: ML Detection (LSTM)  │
│   - Behavior pattern learning   │
│   - Correntropy features        │
│   - Online sliding window       │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   Layer 3: Real-time (RCP-f)    │
│   - Distance-based filtering    │
│   - Convergence guarantee       │
│   - O(n log n) complexity       │
└─────────────────────────────────┘
```

### 2. Novel Correntropy Features

Extended LSTM input from 7D to 10D by adding Maximum Correntropy Criterion features:
- `avg_correntropy`: Average similarity to neighbors
- `min_correntropy`: Minimum similarity (outlier detection)
- `std_correntropy`: Similarity variance

**Theory**: MCC captures all even-order moments, providing richer statistical information than Euclidean distance.

### 3. Integration of Data-Driven Method

Successfully integrated ℓ1 optimization approach from:
> Yan Jiaqi et al., "Secure Data Reconstruction: A Direct Data-Driven Approach"

Demonstrated complementarity with model-driven (RCP-f) and learning-driven (LSTM) methods.

---

## 📖 Documentation

### For Researchers
- **[RESEARCH_FRAMEWORK_SUMMARY.md](RESEARCH_FRAMEWORK_SUMMARY.md)** - Complete research framework overview
- **[organized/docs/RESEARCH_REPORT.md](organized/docs/RESEARCH_REPORT.md)** - Detailed research report
- **[organized/experiments/README.md](organized/experiments/README.md)** - Experiment documentation

### For Developers
- **[organized/README.md](organized/README.md)** - Code structure and usage
- **[organized/docs/CORRECT_METHOD_EXPLANATION.md](organized/docs/CORRECT_METHOD_EXPLANATION.md)** - LSTM methodology
- **[organized/docs/CORRENTROPY_FEATURE_SUMMARY.md](organized/docs/CORRENTROPY_FEATURE_SUMMARY.md)** - Feature engineering

### For Deployment
- **[organized/docs/GITHUB_PUSH_GUIDE.md](organized/docs/GITHUB_PUSH_GUIDE.md)** - GitHub deployment guide

---

## 🎓 Academic Context

### System Model
- **Multi-agent system**: 8 heterogeneous cart-pendulum agents
- **Control objective**: Cooperative output regulation
- **Communication**: Undirected graph topology
- **Byzantine model**: f=1 malicious agent with arbitrary behavior

### Evaluation Metrics
- **Performance recovery rate**: (baseline_error / defense_error) × 100%
- **Detection accuracy**: LSTM classification performance
- **Computational overhead**: Time complexity analysis
- **Convergence rate**: Tracking error over time

---

## 📚 Key References

1. **Yan Jiaqi et al.** - "Secure Data Reconstruction: A Direct Data-Driven Approach"
   - Source of ℓ1 optimization method and Hankel matrix approach

2. **Luan et al. (2025)** - "Maximum Correntropy Criterion-Based Federated Learning"
   - Inspiration for Correntropy features in Byzantine detection

3. **Lamport et al. (1982)** - "The Byzantine Generals Problem"
   - Foundational Byzantine fault tolerance theory

---

## 🛠️ Technical Stack

- **Language**: Python 3.11
- **Deep Learning**: PyTorch
- **Numerical Computing**: NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Version Control**: Git

---

## 📧 Contact & Contribution

**Author**: liziyu
**Institution**: Nanyang Technological University (NTU)
**GitHub**: https://github.com/liziyu6666/NTU-Dissertation

For questions, issues, or collaboration:
- Open an issue on GitHub
- Check documentation in `organized/docs/`
- Review experiment results in `organized/results/`

---

## 📄 License

This project is part of academic research at NTU. Please cite appropriately if using this code for research purposes.

---

## 🎉 Acknowledgments

Special thanks to:
- **Yan Jiaqi et al.** for the data-driven secure reconstruction framework
- **Research advisors** for guidance and feedback
- **Claude Code** for development assistance

---

*Last Updated: 2025-10-30*
*Version: 1.0*
*Status: Complete and ready for paper submission*
