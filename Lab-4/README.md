# Lab 4: CNN Architectures for Imbalanced Image Classification

## Overview

This lab focuses on designing and implementing CNN architectures for handling imbalanced image classification across multiple benchmark datasets. The lab addresses the critical challenge of class imbalance in real-world computer vision applications through both data-level and algorithm-level techniques.

## Problem Statement

Design and implement CNN architectures to perform multi-class image classification on imbalanced datasets, employing various techniques to handle class imbalance and comparing different architectural approaches.

## Datasets

This lab uses **TWO** imbalanced datasets:

### 1. Flower Recognition Dataset
- **Classes**: 5 flower classes (daisy, dandelion, rose, sunflower, tulip)
- **Imbalance Strategy**: Custom sampling to create imbalanced distribution (100:500:200:50:150)
- **Source**: Kaggle - Flowers Recognition Dataset

### 2. CIFAR-10 (Imbalanced Version)
- **Classes**: 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Imbalance Strategy**: Long-tailed distribution with 100:1 imbalance ratio
- **Source**: PyTorch/TorchVision built-in dataset

## Seven Problem Statements Covered

### Problem Statement 1: Architecture Design Focus
- Custom CNN architecture design
- Architecture justification based on dataset characteristics
- Layer design (number, filter sizes, kernel sizes, activations)
- Regularization techniques (Dropout, Batch Normalization, L2)
- Impact analysis of dataset imbalance on performance

### Problem Statement 2: Imbalanced Dataset Handling

**Data-Level Techniques:**
- Random Oversampling of minority classes
- Random Undersampling of majority classes
- SMOTE (Synthetic Minority Oversampling Technique)
- Targeted data augmentation for minority classes

**Algorithm-Level Techniques:**
- Class weighting in loss function
- Cost-sensitive learning
- Threshold adjustment during inference

**Evaluation:**
- Training convergence and stability
- Class-wise accuracy (especially minority classes)
- Macro/Micro F1-scores

### Problem Statement 3: Comparative Architecture Analysis

**Architectures Compared:**
- EfficientNet-B0
- ResNet-50

**Comparison Metrics:**
- Overall accuracy and Top-k accuracy
- Precision, Recall, F1-score (class-wise and macro-averaged)
- Confusion matrix analysis
- Computational cost (FLOPs, parameters, inference time)
- ROC-AUC and PR-AUC curves
- Robustness to imbalance (G-Mean, Balanced Accuracy)

### Problem Statement 4: Loss Function & Optimization Challenge

**Loss Functions:**
- Cross-Entropy Loss (baseline)
- Weighted Cross-Entropy Loss
- Focal Loss (γ = 0.5, 1, 2, 5)
- Class-Balanced Loss
- Label Smoothing Cross-Entropy

**Optimizers:**
- SGD (with/without momentum)
- Adam
- AdamW (Adam with weight decay)
- RMSProp

**Analysis:**
- Convergence speed and training curves
- Minority class recognition improvement
- Overfitting behavior and generalization
- Learning rate scheduling impact

### Problem Statement 5: Feature Representation & Visualization

**Visualization Techniques:**
- t-SNE (t-distributed Stochastic Neighbor Embedding)
- PCA (Principal Component Analysis)
- UMAP (Uniform Manifold Approximation and Projection)
- Grad-CAM (Class Activation Maps)

**Analysis:**
- Feature clustering quality
- Inter-class separation and intra-class compactness
- Minority class representation in feature space
- Decision boundary visualization

### Problem Statement 6: Generalization & Transfer Learning

**Transfer Learning Experiments:**
- Flower dataset → CIFAR-10
- Pre-trained ImageNet weights → Both datasets

**Analysis:**
- Transferability of learned features
- Performance degradation across domains
- Fine-tuning vs feature extraction
- Impact of dataset complexity and domain shift

### Problem Statement 7: Error Analysis & Improvement

**Failure Analysis:**
- Identify frequently failing classes
- Confusion patterns between similar classes
- Error correlation with class imbalance ratios
- Visualization of misclassified samples
- Proposed improvements

## Implementation Details

### Data Preprocessing
- Image resizing and normalization
- Data augmentation (rotation, flip, zoom, color jitter)
- Custom samplers for imbalanced datasets

### Training Configuration
- Batch size: 64 (adjustable based on memory)
- Epochs: 50 (with early stopping)
- Learning rate: 0.001 (with scheduling)
- Device: CUDA (if available) or CPU

### Evaluation Metrics
- Standard metrics: Accuracy, Precision, Recall, F1-score
- Imbalanced-specific: Balanced Accuracy, G-Mean
- Visualization: Confusion Matrix, ROC curves, PR curves

## Key Features

1. **Comprehensive Imbalance Handling**: Multiple techniques at data and algorithm levels
2. **Extensive Evaluation**: Standard and imbalance-specific metrics
3. **Multiple Architectures**: Comparison of modern CNN architectures
4. **Advanced Loss Functions**: Focal Loss, Class-Balanced Loss, etc.
5. **Feature Visualization**: t-SNE, PCA, UMAP, Grad-CAM
6. **Transfer Learning**: Cross-dataset and pre-trained model experiments
7. **Detailed Analysis**: Error analysis and improvement proposals

## Expected Results

### Performance Expectations
- Baseline accuracy: 60-70% on imbalanced datasets
- With imbalance handling: 75-85% overall, improved minority class recognition
- ResNet expected to outperform custom architectures
- Focal Loss expected to improve minority class performance

### Visualization Expectations
- Clear cluster separation in t-SNE/UMAP plots
- Minority classes may show tighter clusters with proper handling
- Grad-CAM should highlight relevant features for each class

## Usage

```bash
cd Lab-4
jupyter notebook Lab-4.ipynb
```

Run all cells sequentially. The notebook will:
1. Download and prepare imbalanced datasets
2. Implement and train multiple architectures
3. Apply various imbalance handling techniques
4. Generate comprehensive evaluation metrics
5. Create visualizations (t-SNE, confusion matrices, ROC curves)
6. Perform transfer learning experiments
7. Conduct error analysis

## Dependencies

Core dependencies (see main `requirements.txt`):
- torch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- scikit-learn >= 0.24.0
- seaborn >= 0.11.0
- tqdm >= 4.60.0
- pandas >= 1.2.0

Additional dependencies for Lab-4:
- imbalanced-learn >= 0.8.0
- umap-learn >= 0.5.0 (optional, visualization will skip if not available)

## Files

- `Lab-4.ipynb`: Main Jupyter notebook with all implementations
- `README.md`: This file
- `IMPLEMENTATION_SUMMARY.md`: Detailed summary of implementations
- `QUICKSTART.md`: Quick guide to run the lab

## Notes

- Training on imbalanced datasets requires careful monitoring of class-wise metrics
- GPU highly recommended for faster training (especially for ResNet and EfficientNet)
- Adjust hyperparameters based on available computational resources
- Some experiments may take several hours to complete
- Consider running subset of experiments for initial exploration

## Dataset Links

1. **Flower Recognition**: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
2. **CIFAR-10**: Available via `torchvision.datasets.CIFAR10`

## References

1. Lin et al. (2017) - Focal Loss for Dense Object Detection
2. Cui et al. (2019) - Class-Balanced Loss Based on Effective Number of Samples
3. He et al. (2016) - Deep Residual Learning for Image Recognition
4. Tan & Le (2019) - EfficientNet: Rethinking Model Scaling for CNNs
5. Chawla et al. (2002) - SMOTE: Synthetic Minority Over-sampling Technique
6. Selvaraju et al. (2017) - Grad-CAM: Visual Explanations from Deep Networks
7. McInnes et al. (2018) - UMAP: Uniform Manifold Approximation and Projection

## Author

**Nilang Bhuva**  
Admission Number: U23AI047  
Program: Artificial Intelligence (AI)  
Year: 3rd Year
