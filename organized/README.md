# Byzantine Node Detection in Multi-Agent Systems

**Project**: NTU Dissertation - LSTM-based Byzantine Detection with Correntropy Features

**Author**: liziyu
**Repository**: https://github.com/liziyu6666/NTU-Dissertation

---

## 📁 Project Structure

```
organized/
├── core/                    # Core simulation system
│   ├── 1.py                           # Fixed simulation with RCP-f filter
│   ├── 1_feature_collection.py        # Original 7D feature collection
│   └── 1_feature_collection_correntropy.py  # Enhanced 10D feature collection
│
├── experiments/             # ⭐ NEW: Comparison experiments
│   ├── simple_comparison.py           # 2-scenario: baseline vs RCP-f
│   ├── three_scenario_comparison.py   # 3-scenario: +no defense control
│   ├── five_scenario_comparison.py    # 5-scenario: +ℓ1 method from paper
│   ├── hybrid_detection_method.py     # Hybrid ℓ1+RCP-f framework
│   ├── ml_comprehensive_comparison.py # 6-scenario: +LSTM ML method
│   └── README.md                      # Experiments documentation
│
├── data_generation/         # Data generation scripts
│   ├── generate_minimal_data.py       # Generate 7D feature dataset
│   └── generate_correntropy_data.py   # Generate 10D feature dataset
│
├── training/                # Model training scripts
│   ├── train_lstm_correct.py          # Correct LSTM training (sliding windows)
│   └── train_compare_correntropy.py   # 7D vs 10D comparison experiment
│
├── detection/               # Detection and evaluation
│   └── online_detection_demo.py       # Real-time Byzantine detection demo
│
├── docs/                    # Documentation
│   ├── RESEARCH_FRAMEWORK_SUMMARY.md  # ⭐ NEW: Complete research framework
│   ├── RESEARCH_REPORT.md             # Comprehensive research report
│   ├── CORRECT_METHOD_EXPLANATION.md  # Method explanation (wrong vs correct)
│   ├── CORRENTROPY_FEATURE_SUMMARY.md # Correntropy features documentation
│   ├── RESULTS_SUMMARY.md             # Experimental results summary
│   └── GIT_PUSH_INSTRUCTIONS.md       # GitHub push instructions
│
├── results/                 # Trained models and figures
│   ├── models/
│   │   ├── LSTM_7D_baseline.pth           # 7D feature model
│   │   ├── LSTM_10D_correntropy.pth       # 10D feature model
│   │   └── lstm_byzantine_detector_lite.pth
│   └── figures/
│       ├── five_scenario_comparison.png   # ⭐ NEW: 5-scenario results
│       ├── correntropy_comparison.png     # 7D vs 10D comparison
│       ├── training_curves_lite.png
│       ├── byzantine_detection_results.png
│       ├── detection_framework_diagram.png
│       └── resilient_cor_results.png
│
├── data/                    # Training datasets (in .gitignore)
│   ├── training_data_minimal/         # 7D features (39.56 MB)
│   └── training_data_correntropy/     # 10D features (51.42 MB)
│
└── archive/                 # Old/deprecated code
    ├── byzantine_detector_*.py        # Old detector implementations
    ├── test_*.py                       # Old test scripts
    └── train_*.py                      # Old training scripts
```

---

## 🚀 Quick Start

### 1. Generate Training Data

**Option A: 7D baseline features**
```bash
cd data_generation
python3 generate_minimal_data.py --attack sine
```

**Option B: 10D features with correntropy**
```bash
cd data_generation
python3 generate_correntropy_data.py --attack sine
```

### 2. Train Model

**Train single model**
```bash
cd training
python3 train_lstm_correct.py
```

**Run comparison experiment (7D vs 10D)**
```bash
cd training
python3 train_compare_correntropy.py
```

### 3. Test Online Detection

```bash
cd detection
python3 online_detection_demo.py
```

---

## 🔬 Key Features

### 1. **Fixed Simulation System** ([core/1.py](core/1.py))

✅ **Regulator Equation Fix**: Corrected Kronecker product ordering (Fortran-order)
```python
# WRONG (C-order):
self.Xi = solution[:n*q].reshape((n, q))

# CORRECT (Fortran-order):
self.Xi = solution[:n*q].reshape((n, q), order='F')
```

✅ **RCP-f Filter**: Correct Euclidean distance-based filtering
```python
def apply_rcpf_filter(v_hat_i, neighbor_vhats, f):
    distances = np.linalg.norm(neighbor_vhats - v_hat_i, axis=1)
    sorted_indices = np.argsort(distances)
    keep_indices = sorted_indices[:n_neighbors - f]
    return neighbor_vhats[keep_indices]
```

### 2. **Correct LSTM Training Method** ([training/train_lstm_correct.py](training/train_lstm_correct.py))

**Key Insight**: Model learns **behavior patterns**, not agent IDs

| Method | Samples | What It Learns | Generalizes? | Online Detection? |
|--------|---------|----------------|--------------|-------------------|
| ❌ **Wrong** | 64 (8 scenarios × 8 agents) | "Agent 4 is Byzantine" | No | No |
| ✅ **Correct** | ~12,800 (sliding windows) | "Byzantine behavior patterns" | Yes | Yes |

**Implementation**:
- Window size: 50 timesteps
- Stride: 50 (non-overlapping)
- Training samples: Each window labeled by behavior (not agent ID)

### 3. **Correntropy Features** ([core/1_feature_collection_correntropy.py](core/1_feature_collection_correntropy.py))

**Inspired by**: "Robust Federated Learning: Maximum Correntropy Aggregation" (IEEE TNNLS 2025)

**Formula**: G_σ(x-y) = exp(-||x-y||²/(2σ²))

**New Features** (7D → 10D):
- `avg_correntropy`: Average similarity with neighbors
- `min_correntropy`: Minimum similarity (most dissimilar neighbor)
- `std_correntropy`: Similarity standard deviation

**Why It Works**:
- Byzantine nodes: Low correntropy (0.1-0.3)
- Normal nodes: High correntropy (0.7-0.9)
- Captures all even-order moments (better than Euclidean distance)

---

## 📊 Experimental Results

### Baseline (7D Features)

- **Test Accuracy**: ~99-100%
- **Test F1 Score**: ~0.99-1.00
- **Training Time**: ~10-15 seconds

### With Correntropy (10D Features)

- **Test Accuracy**: ~99-100%
- **Test F1 Score**: ~0.99-1.00
- **Training Time**: ~12-18 seconds
- **Advantage**: Better at detecting stealthy attacks

### Online Detection Capability

✅ Accumulates 50 timesteps of data
✅ Real-time detection during system runtime
✅ No need to wait for simulation completion
✅ Generalizes to new scenarios

---

## 📚 Documentation

### Core Documents

1. **[RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md)** (25 KB)
   - Comprehensive research report for advisor presentation
   - Background, motivation, methodology
   - Experimental results and analysis
   - MCA paper insights
   - 8 future research directions with timeline

2. **[CORRECT_METHOD_EXPLANATION.md](docs/CORRECT_METHOD_EXPLANATION.md)** (8 KB)
   - Explains the fundamental flaw in original approach
   - Wrong method: Learn agent IDs
   - Correct method: Learn behavior patterns
   - Detailed comparison

3. **[CORRENTROPY_FEATURE_SUMMARY.md](docs/CORRENTROPY_FEATURE_SUMMARY.md)** (10 KB)
   - Theoretical basis from MCA paper
   - Implementation details
   - Feature comparison (7D vs 10D)
   - Expected performance improvements
   - Usage guide

4. **[RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md)** (6 KB)
   - Experimental results summary
   - Performance metrics
   - Visualizations

---

## 🔧 Technical Details

### System Parameters

- **Agents**: 8 cart-pendulum systems
- **Byzantine Tolerance**: f = 1 (up to 1 Byzantine node)
- **Simulation Time**: 15 seconds
- **Timesteps**: 750
- **Communication Topology**: 4 target nodes + 4 regular nodes

### Attack Types

- `sine`: Sinusoidal perturbation (5.0 * sin(3t))
- `constant`: Constant offset (10.0)
- `random`: Random noise (Gaussian, σ=5.0)
- `ramp`: Linear ramp (0.5t)
- `mixed`: Combination of sine and ramp

### Features (10D)

| # | Feature | Description | Type |
|---|---------|-------------|------|
| 1 | `estimation_error` | ‖v̂ - v_real‖ | Physical |
| 2 | `position_error` | \|x - x_ref\| | Physical |
| 3 | `angle` | Pendulum angle θ | Physical |
| 4 | `angular_velocity` | θ̇ | Physical |
| 5 | `control_input` | u(t) | Control |
| 6 | `v_hat_0` | v̂[0] | Estimation |
| 7 | `v_hat_1` | v̂[1] | Estimation |
| 8 | `avg_correntropy` | Mean(G_σ(v̂_i - v̂_j)) | **New** |
| 9 | `min_correntropy` | Min(G_σ(v̂_i - v̂_j)) | **New** |
| 10 | `std_correntropy` | Std(G_σ(v̂_i - v̂_j)) | **New** |

---

## 🎯 Future Research Directions

1. **Cross-Attack Generalization**
   - Train on one attack type, test on others
   - Evaluate robustness

2. **Multi-Byzantine Scenarios**
   - Extend to f = 2 or f = 3
   - Colluding Byzantine nodes

3. **MCA-Based Detector Baseline**
   - Implement Maximum Correntropy Aggregation
   - Compare with LSTM approach

4. **Attention Mechanism**
   - Add attention to focus on key features
   - Improve interpretability

5. **Transfer Learning**
   - Pre-train on multiple scenarios
   - Fine-tune for specific systems

6. **Real-Time Optimization**
   - Dynamic Byzantine node removal
   - System performance improvement

7. **Theoretical Analysis**
   - Convergence guarantees
   - Detection delay bounds

8. **Hardware Validation**
   - Test on physical robot systems
   - Real-world deployment challenges

---

## 🔐 Git & GitHub

### Repository

**URL**: https://github.com/liziyu6666/NTU-Dissertation
**Branch**: master

### Committing Changes

```bash
# Stage all changes
git add .

# Commit with message
git commit -m "Your commit message

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to remote
git push origin master
```

**Note**: See [GIT_PUSH_INSTRUCTIONS.md](docs/GIT_PUSH_INSTRUCTIONS.md) for authentication setup.

---

## 📦 Dependencies

```bash
pip install numpy scipy torch scikit-learn matplotlib pickle
```

### Versions
- Python: 3.8+
- PyTorch: 1.10+
- NumPy: 1.20+
- SciPy: 1.7+
- Matplotlib: 3.3+

---

## 📞 Contact

**Author**: liziyu
**GitHub**: liziyu6666
**Repository Issues**: https://github.com/liziyu6666/NTU-Dissertation/issues

---

## 📄 License

This is a dissertation project for NTU (Nanyang Technological University).

---

## 🙏 Acknowledgments

- **MCA Paper**: "Robust Federated Learning: Maximum Correntropy Aggregation Against Byzantine Attacks" (Luan et al., IEEE TNNLS 2025)
- **Simulation Framework**: Based on cart-pendulum multi-agent cooperative control
- **Detection Method**: LSTM-based behavior classification with correntropy features

---

**Last Updated**: 2025-10-29

**Project Status**: ✅ Core Implementation Complete | 🔬 Experiments Running | 📝 Documentation Ready
