# Lab-4 Quick Start Guide

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Optional dependencies for enhanced features:**
   ```bash
   pip install umap-learn  # For UMAP visualization
   ```

## Running the Notebook

### Option 1: Run All Cells
```bash
jupyter notebook Lab-4.ipynb
# Then: Cell → Run All
```

### Option 2: Execute Specific Problem Statements

The notebook is organized into sections. You can run them independently:

1. **Setup** (Always run first)
   - Cells 1-2: Imports and device configuration

2. **PS1: Architecture Design** (Cells 3-7)
   - Creates imbalanced CIFAR-10 dataset
   - Defines Custom CNN architecture

3. **PS2: Imbalanced Handling** (Cells 8-10)
   - Data-level techniques
   - Loss functions (Focal, Class-Balanced, Label Smoothing)
   - Helper functions

4. **PS3: Architecture Comparison** (Cells 11-16)
   - ResNet-18, EfficientNet-B0
   - Training and evaluation
   - Comparative visualizations

5. **PS4: Loss & Optimizer Experiments** (Cells 17-20)
   - 4 loss functions
   - 4 optimizers
   - Convergence analysis

6. **PS5: Feature Visualization** (Cells 21-26)
   - t-SNE, PCA, UMAP
   - Grad-CAM
   - Clustering quality

7. **PS6: Transfer Learning** (Cells 27-29)
   - Fine-tuning strategies
   - Comparison with scratch training

8. **PS7: Error Analysis** (Cells 30-34)
   - Misclassification analysis
   - Confidence calibration
   - Improvement proposals

9. **Summary** (Cells 35-37)
   - Aggregate results
   - Key findings
   - Conclusions

## Expected Runtime

On CPU:
- **Full notebook**: ~4-6 hours
- **Single problem statement**: ~30-60 minutes

On GPU (CUDA):
- **Full notebook**: ~1-2 hours
- **Single problem statement**: ~10-20 minutes

## Tips for Faster Execution

1. **Reduce epochs:**
   ```python
   num_epochs = 5  # Instead of 15
   num_epochs_loss = 5  # Instead of 12
   num_epochs_ft = 5  # Instead of 10
   ```

2. **Use smaller batch size for CPU:**
   ```python
   batch_size = 64  # Instead of 128
   ```

3. **Limit feature extraction samples:**
   ```python
   features, labels = extract_features(model, test_loader, device, max_samples=1000)
   ```

4. **Skip optional visualizations:**
   - Comment out UMAP section if not installed
   - Reduce number of Grad-CAM examples

## Output Files

All visualizations are saved to the current directory:
- `cifar10_class_distribution.png`
- `architecture_training_comparison.png`
- `architecture_metrics_comparison.png`
- `loss_optimizer_comparison.png`
- `tsne_visualization.png`
- `pca_visualization.png`
- `gradcam_visualization.png`
- `transfer_learning_comparison.png`
- `error_analysis_confusion_matrix.png`
- `misclassified_examples.png`
- `confidence_analysis.png`
- `comprehensive_results_heatmap.png`
- `top5_radar_chart.png`

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
batch_size = 32

# Clear cache between experiments
torch.cuda.empty_cache()
```

### Import Errors
```bash
# Install missing packages
pip install imbalanced-learn
pip install umap-learn  # Optional
```

### Slow t-SNE
```python
# Use PCA initialization
tsne = TSNE(n_components=2, init='pca', random_state=42)

# Reduce perplexity
tsne = TSNE(n_components=2, perplexity=15, random_state=42)
```

## Customization

### Use Different Dataset
Replace CIFAR-10 with your own:
```python
# Custom dataset
train_dataset = YourDataset(transform=transform_train)
test_dataset = YourDataset(transform=transform_test)
```

### Different Imbalance Ratio
```python
cifar_train, cifar_test, counts = create_imbalanced_cifar10(imbalance_ratio=50)
```

### Add Your Own Model
```python
class YourCNN(nn.Module):
    # Your architecture here
    pass

# Add to comparison
your_model = YourCNN().to(device)
```

## Key Results to Expect

1. **Architecture Comparison**: ResNet-18 and EfficientNet typically outperform Custom CNN
2. **Loss Functions**: Focal Loss and Class-Balanced Loss improve minority class performance
3. **Transfer Learning**: Pretrained models show 5-10% accuracy improvement
4. **Error Analysis**: Visually similar classes (cat/dog) show higher confusion

## Learning Objectives

By completing this lab, you will understand:
- ✓ How to handle imbalanced image datasets
- ✓ Specialized loss functions for imbalanced learning
- ✓ Transfer learning benefits and strategies
- ✓ Feature visualization techniques
- ✓ Comprehensive model evaluation and error analysis
- ✓ Practical techniques for production systems

## Support

For issues or questions:
1. Check the error message and troubleshooting section
2. Verify all dependencies are installed
3. Ensure adequate memory (8GB+ RAM recommended)
4. Try reducing batch size or epochs

Happy Learning! 🎓
