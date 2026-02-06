# Lab-3 Quick Start Guide

## Prerequisites
Ensure you have Python 3.7+ installed.

## Installation

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and TorchVision
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (for t-SNE)
- Jupyter Notebook
- tqdm (for progress bars)

### Step 2: Verify Installation (Optional)
```bash
python3 verify_lab3.py
```

Expected output: "✅ Notebook structure verification PASSED"

## Running the Notebook

### Option 1: Jupyter Notebook
```bash
cd Lab-3
jupyter notebook Lab-3.ipynb
```

### Option 2: Jupyter Lab
```bash
cd Lab-3
jupyter lab Lab-3.ipynb
```

### Option 3: VS Code
1. Open VS Code
2. Install "Jupyter" extension
3. Open `Lab-3/Lab-3.ipynb`
4. Select Python kernel
5. Run cells

## Notebook Execution

### Quick Demo (Recommended for First Run)
The notebook is structured to run a subset of architectures for demonstration:
- Part 1: Trains 4 architectures (LeNet-5, VGGNet, ResNet-50, MobileNet)
- Part 2: Trains 3 specific configurations (VGGNet+BCE, AlexNet+Focal, ResNet+ArcFace)
- Part 3: Creates t-SNE visualizations

**Estimated time:**
- With GPU: 15-30 minutes
- Without GPU: 1-2 hours

### Full Training (All 8 Architectures)
To train all architectures, modify the Part 1 section to include:
- ResNet-100
- EfficientNet
- InceptionV3
- Additional AlexNet runs

**Estimated time:**
- With GPU: 1-2 hours
- Without GPU: 4-6 hours

## Expected Outputs

### 1. Console Output
- Dataset download progress
- Training progress bars for each epoch
- Accuracy and loss metrics
- Summary tables

### 2. Visualizations (PNG files)
- `part1_architecture_comparison.png`: Comparison of 4 architectures
- `part2_loss_optimizer_comparison.png`: Loss function comparison
- `part3_tsne_visualization.png`: Feature clustering visualization

### 3. Data Downloads
- CIFAR-10 dataset (~170MB) will be downloaded to `./data/` folder on first run

## Troubleshooting

### Out of Memory (GPU)
**Problem:** CUDA out of memory error

**Solution:**
1. Reduce batch size in the data loader:
   ```python
   train_loader = DataLoader(..., batch_size=64)  # Instead of 128
   ```
2. Train one model at a time
3. Use CPU instead (set `device = 'cpu'`)

### Out of Memory (CPU)
**Problem:** System runs out of RAM

**Solution:**
1. Reduce batch size to 32 or 16
2. Train smaller architectures first (LeNet-5, MobileNet)
3. Reduce number of samples for t-SNE (default is 1000)

### Slow Training
**Problem:** Training takes too long

**Solution:**
1. Reduce number of epochs:
   ```python
   train_model(..., epochs=3)  # Instead of 5, 10, 15, 20
   ```
2. Use GPU if available
3. Train only selected architectures
4. Use smaller subset of data for quick testing

### Import Errors
**Problem:** ModuleNotFoundError

**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### CIFAR-10 Download Fails
**Problem:** Network error downloading dataset

**Solution:**
1. Download manually from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
2. Extract to `./data/cifar-10-batches-py/`
3. Re-run notebook

## Tips for Best Results

1. **Use GPU**: Training is much faster on GPU
   ```python
   # Check if GPU is available
   print(torch.cuda.is_available())
   ```

2. **Start Small**: Run with reduced epochs first to verify everything works
   
3. **Monitor Progress**: Watch the progress bars and accuracy metrics

4. **Save Checkpoints**: For long training, consider saving model checkpoints

5. **Compare Results**: Focus on comparing relative performance, not absolute accuracy

## Understanding the Results

### Part 1: Architecture Comparison
- **LeNet-5**: Lowest accuracy (simple architecture)
- **AlexNet**: Good accuracy with moderate complexity
- **VGGNet**: High accuracy but many parameters
- **ResNet-50**: Best accuracy with skip connections
- **MobileNet**: Good accuracy with efficiency

### Part 2: Loss Function Impact
- **BCE**: Standard baseline
- **Focal Loss**: Better for imbalanced data
- **ArcFace**: Best feature discrimination

### Part 3: t-SNE Visualization
- **BCE clusters**: May have some overlap
- **ArcFace clusters**: Tighter, more separated

## Customization

### Train Different Architecture
```python
model = InceptionV3().to(device)  # Change to any architecture
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
results = train_model(model, train_loader, test_loader, 
                     criterion, optimizer, 5, device, 'InceptionV3')
```

### Try Different Optimizer
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# or
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### Change Learning Rate
```python
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR
```

### Use Different Dataset
```python
# For MNIST (easier, faster)
train_dataset = torchvision.datasets.MNIST(...)

# For Fashion-MNIST
train_dataset = torchvision.datasets.FashionMNIST(...)
```

## Next Steps

After completing Lab-3:
1. Experiment with different hyperparameters
2. Try transfer learning with pretrained models
3. Implement custom architectures
4. Add learning rate scheduling
5. Implement early stopping
6. Try different data augmentation strategies

## Additional Resources

- PyTorch Documentation: https://pytorch.org/docs/
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Original Papers:
  - LeNet: http://yann.lecun.com/exdb/lenet/
  - AlexNet: https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
  - ResNet: https://arxiv.org/abs/1512.03385
  - MobileNet: https://arxiv.org/abs/1704.04861

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure Python version is 3.7+
4. Check available memory (RAM/GPU)
5. Review error messages carefully

---

**Happy Learning!** 🚀
