# Deep Learning Lab Assignments

**Student Information:**
- **Name:** Nilang Bhuva
- **Admission Number:** U23AI047
- **Year:** 3rd Year
- **Program:** Artificial Intelligence (AI)

## 📚 Repository Overview

This repository contains my Deep Learning Lab assignments for the AI program. Each lab focuses on different aspects of deep learning, from fundamental tensor operations to building and training neural networks.

## 📂 Repository Structure

```
DL-LAb/
├── Lab-1/                      # Lab 1: PyTorch Fundamentals
│   └── Lab1.ipynb             # PyTorch tensors, operations, and basics
├── Lab-2/                      # Lab 2: Neural Networks on MNIST
│   ├── Lab-2.ipynb            # CNN and MLP implementations
│   └── data/                  # MNIST dataset
│       └── MNIST/
├── Lab-3/                      # Lab 3: CNN Architectures Comparison
│   └── Lab-3.ipynb            # Multiple CNN architectures and loss functions
└── README.md                   # This file
```

## 📝 Lab Assignments

### Lab 1: Introduction to PyTorch Tensors and Basic Operations

**Topics Covered:**
- PyTorch tensor initialization methods and data types
- Tensor operations (arithmetic, broadcasting, indexing, reshaping)
- Automatic differentiation using Autograd
- Linear algebra operations with PyTorch
- Implementation of AND & OR gates using Perceptron

**Key Concepts:**
- Creating tensors from lists, zeros, ones, random values
- Tensor arithmetic and matrix operations
- Gradient computation with autograd
- Basic perceptron implementation

**File:** `Lab-1/Lab1.ipynb`

---

### Lab 2: Convolutional Neural Networks and Multi-Layer Perceptrons

**Topics Covered:**
- MNIST dataset loading and preprocessing
- Convolutional Neural Network (CNN) implementation
- Multi-Layer Perceptron (MLP) implementation
- Training and testing functions
- Model evaluation and comparison

**Key Concepts:**
- CNN architecture with convolutional and pooling layers
- MLP (fully connected) architecture
- Adam optimizer implementation
- Model training and validation on MNIST dataset

**File:** `Lab-2/Lab-2.ipynb`

---

### Lab 3: Comparative Analysis of Different CNN Architectures

**Topics Covered:**
- Implementation of landmark CNN architectures (LeNet-5, AlexNet, VGGNet, ResNet-50, ResNet-100, EfficientNet, InceptionV3, MobileNet)
- CIFAR-10 dataset preprocessing and augmentation
- Advanced loss functions (BCE, Focal Loss, ArcFace)
- Optimizer comparison (Adam, SGD with momentum)
- Feature visualization with t-SNE
- Comparative analysis of architecture performance

**Key Concepts:**
- Deep CNN architectures and their evolution
- Residual connections and skip connections
- Depthwise separable convolutions
- Inception modules
- Loss function impact on feature learning
- Feature space visualization and clustering

**Part 1:** Train and compare 8 different CNN architectures on CIFAR-10
**Part 2:** Study impact of loss functions (BCE, Focal Loss, ArcFace) and optimizers
**Part 3:** Visualize feature clustering using t-SNE for different loss functions

**File:** `Lab-3/Lab-3.ipynb`


---

## 🛠️ Prerequisites

Before running the labs, ensure you have the following installed:

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- tqdm
- Pandas

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nilangbhuva/DL-LAb.git
   cd DL-LAb
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running Lab Notebooks

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Navigate to the desired lab folder** (Lab-1, Lab-2, or Lab-3) in the Jupyter interface

3. **Open the notebook** (.ipynb file) and run the cells sequentially

### Running Individual Labs

**Lab 1:**
```bash
cd Lab-1
jupyter notebook Lab1.ipynb
```

**Lab 2:**
```bash
cd Lab-2
jupyter notebook Lab-2.ipynb
```

**Lab 3:**
```bash
cd Lab-3
jupyter notebook Lab-3.ipynb
```

## 📊 Lab Results

### Lab 1 Outcomes
- Successfully implemented PyTorch tensor operations
- Demonstrated automatic differentiation capabilities
- Built basic perceptron for logic gates

### Lab 2 Outcomes
- Trained CNN and MLP models on MNIST dataset
- Compared performance of different architectures
- Achieved classification accuracy on handwritten digit recognition

### Lab 3 Outcomes
- Implemented and compared 8 landmark CNN architectures
- Analyzed impact of different loss functions on model performance
- Visualized feature clustering with t-SNE
- Demonstrated superiority of advanced architectures (ResNet) over classical ones (LeNet)

## 🔍 Key Learnings

- Understanding of PyTorch tensor operations and computational graphs
- Implementation of fundamental neural network architectures
- Experience with training deep learning models
- Dataset preprocessing and model evaluation techniques
- Comparative analysis of CNN architectures and their evolution
- Impact of loss functions on feature learning and discrimination
- Feature visualization and clustering techniques

## 📧 Contact

For any queries regarding this repository:
- **Student:** Nilang Bhuva
- **Admission Number:** U23AI047

## 📄 License

This repository is for educational purposes as part of the Deep Learning Lab coursework.

---

**Note:** This repository will be updated as new lab assignments are completed throughout the semester.
