# Lab-2: Deep Learning Experiments

## Overview
This lab explores fundamental concepts in deep learning through systematic experimentation with:
- Activation functions
- Optimizers
- Regularization techniques (Batch Normalization and Dropout)

## Tasks

### Task 1: The Activation Function Challenge
Compare the training loss and accuracy curves when using different activation functions:
- **Sigmoid**: Observe if the model suffers from "vanishing gradients" or slow start
- **Tanh**: Compare its speed to Sigmoid
- **ReLU**: Document why this usually leads to faster convergence

### Task 2: The Optimizer Showdown
Using the best activation function (ReLU), compare different optimizers:
- **SGD (Stochastic Gradient Descent)**: Observe the stability of the loss
- **SGD with Momentum**: Note how it handles "bumps" in the loss landscape
- **Adam**: Observe how quickly it reaches high accuracy compared to basic SGD

### Task 3: Batch Normalization and Dropout
Run specific scenarios to observe the contrast:
- WITHOUT Batch Normalization and Dropout
- Without BN, Dropout layer=0.1
- With BN, Dropout layer=0.25

## Requirements

Install the required dependencies:

```bash
pip install torch torchvision numpy matplotlib pandas jupyter
```

## Running the Notebook

1. Navigate to the Lab-2 directory:
```bash
cd Lab-2
```

2. Start Jupyter Notebook:
```bash
jupyter notebook Lab-2.ipynb
```

3. Run all cells in sequence to:
   - Load the MNIST dataset
   - Define the neural network model
   - Run all 9 experiments
   - View comparison tables and visualizations

## Expected Outputs

The notebook will generate:

1. **Comparison Table**: Shows all experiments with their configurations and final test accuracies
2. **Visualizations**:
   - Task 1: Loss and accuracy curves for different activation functions
   - Task 2: Loss and accuracy curves for different optimizers
   - Task 3: Loss and accuracy curves for different regularization scenarios
   - Comprehensive comparison plots

## Model Architecture

The neural network used in all experiments has the following architecture:
- Input Layer: 784 neurons (28x28 flattened MNIST images)
- Hidden Layer 1: 128 neurons
- Hidden Layer 2: 64 neurons
- Output Layer: 10 neurons (digit classes 0-9)

Configurable components:
- Activation functions: Sigmoid, Tanh, or ReLU
- Batch Normalization: Optional
- Dropout: Configurable rate (0.0, 0.1, 0.25)

## Key Findings

**Activation Functions:**
- ReLU typically performs best, avoiding vanishing gradient problems
- Sigmoid and Tanh can show slower convergence
- ReLU leads to faster training and better accuracy

**Optimizers:**
- Adam converges fastest with adaptive learning rates
- SGD with Momentum handles loss oscillations better than basic SGD
- Adam generally achieves highest accuracy with default hyperparameters

**Regularization:**
- Batch Normalization improves training stability
- Dropout prevents overfitting
- Combining BN with moderate dropout gives best generalization

## Experiment Configuration

All experiments run for:
- **10 epochs**
- **Batch size**: 128
- **Learning rate**: 0.01 for SGD, 0.001 for Adam
- **Momentum**: 0.9 (when used)

## Notes

- The MNIST dataset will be automatically downloaded on first run
- All experiments use the same model architecture for fair comparison
- Results may vary slightly between runs due to random initialization
- GPU acceleration is used if available, otherwise CPU
