# Lab 5: Recurrent Neural Network Architectures for Handwritten Character Recognition

## Overview

This lab designs and implements Recurrent Neural Network (RNN) architectures for handwritten character recognition using the **MNIST** and **EMNIST** datasets. A key insight exploited throughout is that a 28×28 image can be treated as a sequence of 28 time steps, where each time step is a row of 28 pixel values, allowing sequential models to learn spatial patterns.

## Problem Statement

Design and implement RNN-based architectures to perform handwritten digit and character recognition, covering vanilla RNNs, LSTMs, GRUs, bidirectional variants, and CNN-LSTM hybrids, with comprehensive comparative analysis and hyperparameter tuning.

## Datasets

| Dataset | Classes | Training | Test |
|---|---|---|---|
| MNIST | 10 (Digits 0–9) | 60,000 | 10,000 |
| EMNIST-Letters | 26 (A–Z) | 124,800 | 20,800 |

Both datasets are loaded via `torchvision.datasets`. A subset (10,000 train / 2,000 test) is used for rapid experimentation; the full dataset is straightforward to enable by removing the `Subset` wrappers.

## Seven Problem Statements Covered

### Problem Statement 1: Vanilla RNN Implementation
- **VanillaRNNScratch**: Multi-layer RNN built from individual `VanillaRNNCell` (from scratch).
- **VanillaRNNPyTorch**: Uses `nn.RNN` as the building block.
- **Vanishing gradient analysis**: Gradient norms per layer for 1-layer vs 3-layer networks.
- **Row-wise vs column-wise scanning**: Transposing the input to compare scanning directions.
- **Architecture sweep**: 1/2/3 layers × 64/128/256 hidden units; heatmap of validation accuracy.

### Problem Statement 2: LSTM Implementation
- **LSTMModel**: Stacked `nn.LSTM` with configurable dropout.
- **LSTMWithGateCapture**: Custom LSTMCell-based model that records forget/input/cell/output gate activations at each time step.
- Experiments: single vs multi-layer, hidden units (32/64/128/256), dropout (0/0.2/0.3/0.5).
- Comparison with Vanilla RNN.

### Problem Statement 3: GRU Implementation
- **GRUModel**: Stacked `nn.GRU`.
- GRU vs LSTM: accuracy, training time, parameter count.
- FLOPs and memory estimation for RNN / GRU / LSTM.
- Stacked GRU experiments (1/2/3 layers).
- Discussion on when to prefer GRU over LSTM.

### Problem Statement 4: Bidirectional LSTM
- **BiLSTMConcat**: Forward + backward final states concatenated.
- **BiLSTMAverage**: Forward + backward final states averaged.
- **BiGRUModel**: Bidirectional GRU.
- Comparison with unidirectional LSTM; analysis of whether bidirectional processing helps for image sequences.

### Problem Statement 5: CNN + LSTM Hybrid Architecture
- **CNNLSTMModel**: 1D CNN applied to each image row, output fed to LSTM.
- **TimeDistributedCNNLSTM**: 2D CNN applied to non-overlapping row patches, output fed to LSTM.
- **PureCNN** baseline: standard 2D convolutional network.
- Feature map and filter visualization.
- Accuracy, parameter count, and inference time trade-off analysis.

### Problem Statement 6: Hyperparameter Tuning & Regularization
- Learning rate sweep: 0.01, 0.001, 0.0001.
- Batch size sweep: 64, 128, 256.
- Gradient clipping sweep: None, 1.0, 5.0.
- Optimizer comparison: SGD, Adam, RMSprop, AdamW.
- **EarlyStopping** class with patience parameter.
- `ReduceLROnPlateau` learning rate scheduling.
- Dropout regularization analysis.

### Problem Statement 7: Comprehensive Comparative Analysis
- All models trained for 7 epochs; results compiled in a summary DataFrame.
- Training/validation curves for all 7 architectures on one plot.
- Comparison bar charts: accuracy, parameters, time/epoch, inference latency, memory.
- Confusion matrices for 4 representative models.
- **t-SNE** embedding of penultimate-layer features.
- Misclassified sample visualization with per-class accuracy breakdown.
- EMNIST-Letters evaluation (when available).

## Key Features

1. **From-scratch implementation** of `VanillaRNNCell` alongside PyTorch built-in variants.
2. **Gate activation heatmaps** across all 28 time steps for LSTM (input, forget, cell, output gates).
3. **FLOPs estimator** for comparing computational cost across cell types.
4. **EarlyStopping + LR Scheduling** pipeline with `ReduceLROnPlateau`.
5. **CNN feature map visualization** (filters and activation maps).
6. **t-SNE** feature-space visualization for cluster quality assessment.
7. **Inference time measurement** for practical deployment comparisons.

## Architecture Summary

| Model | Key Characteristics |
|---|---|
| Vanilla RNN | Minimal gating; susceptible to vanishing gradients |
| LSTM | 4 gates (forget, input, cell, output); strong long-range memory |
| GRU | 2 gates (reset, update); faster than LSTM, fewer parameters |
| BiLSTM | Processes sequence in both directions; best accuracy on MNIST |
| BiGRU | Bidirectional GRU; good balance of accuracy and speed |
| CNN-LSTM | Spatial features per row via 1D CNN + temporal LSTM |
| TD-CNN-LSTM | Patch-level 2D CNN + LSTM; compact and efficient |

## Expected Results (MNIST, 7 epochs, 10 k train / 2 k test subset)

| Model | ~Val Acc | ~Parameters |
|---|---|---|
| Vanilla RNN | 0.85–0.91 | ~50 K |
| LSTM-1L | 0.92–0.96 | ~100 K |
| GRU | 0.91–0.95 | ~75 K |
| BiLSTM | 0.93–0.97 | ~200 K |
| CNN-LSTM | 0.93–0.97 | ~150 K |

*(Exact values depend on hardware and random seed.)*

## Usage

```bash
cd Lab-5
jupyter notebook Lab-5.ipynb
```

Run all cells sequentially. The notebook will:
1. Download MNIST (and optionally EMNIST) automatically.
2. Train and evaluate all architectures.
3. Generate all visualizations inline.

## Dependencies

All required packages are listed in the repository-level `requirements.txt`:
- `torch >= 2.0.0`
- `torchvision >= 0.10.0`
- `numpy >= 1.19.0`
- `matplotlib >= 3.3.0`
- `scikit-learn >= 1.0.1`
- `seaborn >= 0.11.0`
- `pandas >= 1.2.0`
- `tqdm >= 4.60.0`

## Files

- `Lab-5.ipynb` — Main Jupyter notebook with all implementations and analysis.
- `README.md` — This file.

## Dataset Links

1. **MNIST**: http://yann.lecun.com/exdb/mnist/ — via `torchvision.datasets.MNIST`
2. **EMNIST**: https://www.nist.gov/itl/products-and-services/emnist-dataset — via `torchvision.datasets.EMNIST`
3. **Kaggle EMNIST**: https://www.kaggle.com/datasets/crawford/emnist

## References

1. Hochreiter & Schmidhuber (1997) – Long Short-Term Memory
2. Cho et al. (2014) – Learning Phrase Representations using RNN Encoder-Decoder (GRU)
3. Schuster & Paliwal (1997) – Bidirectional Recurrent Neural Networks
4. LeCun et al. (1998) – Gradient-Based Learning Applied to Document Recognition (MNIST)
5. Cohen et al. (2017) – EMNIST: Extending MNIST to Handwritten Letters

## Author

**Nilang Bhuva**  
Admission Number: U23AI047  
Program: Artificial Intelligence (AI)  
Year: 3rd Year
