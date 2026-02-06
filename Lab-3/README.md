# Lab 3: Comparative Analysis of Different CNN Architectures

## Overview

This lab implements and compares landmark Convolutional Neural Network (CNN) architectures across the CIFAR-10 dataset to analyze the impact of network depth, architecture design, loss functions, and optimization strategies on classification accuracy and computational efficiency.

## Problem Statement

### Part 1: Architecture Comparison
Implement, train, and evaluate the following CNN architectures:
- **LeNet-5**: Classical CNN architecture (1998)
- **AlexNet**: Deep CNN that won ImageNet 2012
- **VGGNet**: Very deep networks with small filters
- **ResNet-50**: Residual networks with 50 layers
- **ResNet-100**: Residual networks with 100 layers
- **EfficientNet**: Compound scaling of networks
- **InceptionV3**: Networks with inception modules
- **MobileNet**: Efficient networks for mobile devices

### Part 2: Loss Function and Optimization Study
Compare advanced loss functions and their impact on model convergence:

| Model | Optimizer | Epochs | Loss Function | Training Acc | Testing Acc |
|-------|-----------|--------|---------------|--------------|-------------|
| VGGNet | Adam | 10 | BCE | ? | ? |
| AlexNet | SGD | 20 | Focal Loss | ? | ? |
| ResNet | Adam | 15 | ArcFace | ? | ? |

**Loss Functions Implemented:**
- **Binary Cross-Entropy (BCE)**: Standard binary classification loss adapted for multi-class
- **Focal Loss**: Addresses class imbalance by down-weighting easy examples
- **ArcFace**: Additive angular margin loss for better feature discrimination

### Part 3: Feature Visualization
- Use t-SNE to visualize how different loss functions cluster features
- Compare feature representations between BCE and ArcFace
- Demonstrate the impact of loss functions on feature separation

## Dataset

**CIFAR-10**: 60,000 32x32 color images in 10 classes
- Training samples: 50,000
- Test samples: 10,000
- Classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck

## Implementation Details

### Data Augmentation
- Random horizontal flips
- Random crops with padding
- Normalization using CIFAR-10 statistics

### Training Configuration
- Batch size: 128
- Device: CUDA (if available) or CPU
- Random seed: 42 (for reproducibility)

### Architectures

#### LeNet-5
- 2 convolutional layers
- 3 fully connected layers
- Max pooling
- ReLU activations

#### AlexNet
- 5 convolutional layers
- 3 fully connected layers
- Max pooling and dropout
- ReLU activations

#### VGGNet
- Multiple convolutional blocks with small 3x3 filters
- Max pooling between blocks
- Dense classifier with dropout

#### ResNet-50/100
- Bottleneck residual blocks
- Skip connections
- Batch normalization
- Identity and projection shortcuts

#### EfficientNet (Simplified)
- Efficient scaling
- Depthwise separable convolutions
- Adaptive pooling

#### InceptionV3
- Inception modules with multiple filter sizes
- Parallel convolutional paths
- Concatenation of features

#### MobileNet
- Depthwise separable convolutions
- Efficient for mobile deployment
- Reduced computational cost

## Key Features

1. **Modular Design**: Each architecture is implemented as a separate class
2. **Flexible Training**: Configurable loss functions and optimizers
3. **Comprehensive Evaluation**: Training and testing metrics tracked per epoch
4. **Visualization**: t-SNE plots for feature space analysis
5. **Progress Tracking**: tqdm progress bars for training/testing loops

## Results

The notebook generates:
- Training and testing accuracy curves for each architecture
- Comparative plots showing all architectures together
- Loss function comparison for Part 2
- t-SNE visualization plots for Part 3
- Summary tables with final accuracies

### Expected Observations

**Part 1:**
- ResNet architectures should achieve highest accuracy due to skip connections
- LeNet-5 will have lower accuracy due to simpler architecture
- MobileNet provides good balance between accuracy and efficiency

**Part 2:**
- Focal Loss helps with difficult examples
- ArcFace produces better feature separation
- BCE provides stable baseline performance

**Part 3:**
- ArcFace features form tighter, more separated clusters
- BCE features may show more overlap between classes
- Visual confirmation of loss function impact on learned representations

## Usage

```bash
cd Lab-3
jupyter notebook Lab-3.ipynb
```

Run all cells sequentially. The notebook will:
1. Download CIFAR-10 dataset (if not already present)
2. Train selected architectures
3. Generate comparison plots
4. Create t-SNE visualizations
5. Print summary tables

## Dependencies

All dependencies are listed in the main `requirements.txt`:
- torch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- scikit-learn >= 0.24.0
- seaborn >= 0.11.0
- tqdm >= 4.60.0
- pandas >= 1.2.0

## Notes

- Training all 8 architectures can be time-consuming. The notebook includes a subset for demonstration.
- GPU is highly recommended for faster training.
- Adjust epochs and batch size based on available computational resources.
- The notebook saves plots as PNG files in the Lab-3 directory.

## References

1. LeCun et al. (1998) - LeNet-5
2. Krizhevsky et al. (2012) - AlexNet
3. Simonyan & Zisserman (2014) - VGGNet
4. He et al. (2016) - ResNet
5. Szegedy et al. (2016) - InceptionV3
6. Howard et al. (2017) - MobileNet
7. Tan & Le (2019) - EfficientNet
8. Lin et al. (2017) - Focal Loss
9. Deng et al. (2019) - ArcFace

## Author

**Nilang Bhuva**  
Admission Number: U23AI047  
Program: Artificial Intelligence (AI)  
Year: 3rd Year
