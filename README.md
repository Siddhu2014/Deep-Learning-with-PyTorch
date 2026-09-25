# Deep Learning with PyTorch

A from-first-principles journey through deep learning with PyTorch.

This repository is being built alongside a structured learning process. The goal is not simply to learn PyTorch APIs, but to understand the mathematics, computational model, and design decisions behind modern deep learning.

The approach is:

> **Understand first. Implement second. Automate third.**

PyTorch is used as a tool for expressing concepts that are already understood, rather than as a black box.

---

## Learning Philosophy

The learning process follows three stages:

```text
Understand the concept
        ↓
Implement the mechanism
        ↓
Use PyTorch to automate it

## Learning Path

### 01 — Tensors
**Status:** Completed  
The foundation of PyTorch.  
**Topics covered:**
- Tensor creation
- Tensor dimensions and axes
- Tensor shapes
- Tensor data types
- Tensor devices
- Tensor metadata
- Tensor indexing and slicing
- Tensor views
- Shared storage
- Tensor cloning
- Reshaping
- Tensor operations
- Element-wise operations
- Broadcasting
- Matrix operations
- NumPy interoperability
- CPU and GPU tensors

*Open `01_Tensors/main.ipynb`*

### 02 — Autograd
**Status:** Completed  
Understanding how PyTorch computes gradients automatically.  
**Topics covered:**
- `requires_grad`
- Computational graphs
- `grad_fn`
- `next_functions`
- Branching computational graphs
- `backward()`
- Leaf and non-leaf tensors
- `retain_grad()`
- Gradient accumulation
- `torch.no_grad()`
- `requires_grad_(False)`
- `detach()`
- `clone()` vs `detach()`
- Trainable parameters
- Gradient descent
- Weight and bias
- Vectorized training
- `nn.Linear`
- `MSELoss`
- SGD
- Complete PyTorch training loops

*Open `02_Autograd/main.ipynb`*

### 03 — Neural Networks
**Status:** Completed  
Moving from individual mathematical operations to complete neural network architectures.  
**Topics covered:**
- `nn.Module`
- Multiple layers
- Parameters
- Activation functions
- Forward propagation
- Multi-layer networks
- Nonlinear transformations
- Network width and depth
- Training and validation
- `DataLoader`
- CPU vs GPU for small workloads

*Open `03_Neural_Networks/main.ipynb`*

### 04 — Classification
**Status:** Completed  
Understanding classification from logits through nonlinear decision boundaries.  
**Topics covered:**
- Classification targets and class indices
- Multi-class classifiers
- `CrossEntropyLoss`
- Logits
- `argmax` and class prediction
- Training and validation accuracy
- `model.train()` and `model.eval()`
- `torch.no_grad()`
- Model width and depth experiments
- Run-to-run variation
- Decision-boundary visualization
- Linear vs nonlinear classification
- ReLU and nonlinear decision boundaries
- Two Moons dataset

*Open `04_Classification/main.ipynb`*

### 05 — Computer Vision
**Status:** Completed  
Applying deep learning to image data and understanding convolutional neural networks from first principles.  
**Topics covered:**
- Image tensors
- Image dimensions and channels
- Convolution
- Kernels and filters
- Feature maps
- Parameter sharing
- Stride
- Padding
- Output-size calculations
- Multiple convolutional filters
- Convolutional layers
- ReLU
- Max pooling
- Flattening feature maps
- Fully connected classification layers
- CNN architecture
- CIFAR-10
- Image classification
- Data augmentation
- Training CNNs with PyTorch
- Model evaluation
- Confusion matrices
- Misclassification analysis
- Visual inspection of predictions
- CNN experiments and architecture comparison

**CNN Architecture**  
The main CNN developed in this chapter follows the structure:

```text
Input Image
    │
    ▼
 Conv2D
    │
    ▼
  ReLU
    │
    ▼
 MaxPool
    │
    ▼
 Conv2D
    │
    ▼
  ReLU
    │
    ▼
 MaxPool
    │
    ▼
 Flatten
    │
    ▼
Linear Layer
    │
    ▼
Class Logits
```

The chapter uses CIFAR-10 to connect the underlying convolution operation with an actual trainable image-classification model.  
*Open `05_Computer_Vision/main.ipynb`*  
*Open `05_Computer_Vision/README.md`*

### 06 — Training & Generalization
**Status:** Next  
Going deeper into how neural networks actually learn and generalize.  
**Planned topics:**
- Loss functions
- Optimizers
- Learning rates
- Learning-rate effects
- Training dynamics
- Validation
- Evaluation
- Overfitting
- Underfitting
- Regularization
- Weight decay
- Dropout
- Training diagnostics
- Model capacity
- Generalization
- Data augmentation
- Hyperparameter experiments

The objective is to understand why training succeeds or fails, rather than simply knowing how to run a training loop.

### 07 — Sequence Models
**Status:** Planned  
Understanding models designed for sequential and temporal data.  
**Planned topics:**
- Sequential data
- Sequence representations
- Recurrent neural networks
- Hidden states
- Recurrence
- Unrolling through time
- Vanishing and exploding gradients
- LSTMs
- GRUs
- Sequence modelling
- Sequence classification

The goal is to understand why recurrent architectures were developed and what limitations eventually motivated attention-based architectures.

### 08 — Attention
**Status:** Planned  
Building attention from the underlying mathematical operations.  
**Planned topics:**
- Motivation for attention
- Query, Key, Value
- Similarity
- Attention scores
- Scaling
- Softmax
- Weighted combinations
- Self-attention
- Masking
- Multi-head attention
- Positional information

The objective is to understand attention mathematically before treating it as a PyTorch abstraction.

### 09 — Transformers
**Status:** Planned  
Understanding the architecture behind modern language and multimodal models.  
**Planned topics:**
- Transformer architecture
- Encoder and decoder concepts
- Self-attention
- Multi-head attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Positional encodings
- Causal masking
- Transformer training
- Token representations

The goal is to progress from a manually understandable attention mechanism to a complete Transformer architecture.

### 10 — Embeddings & Language Models
**Status:** Planned  
Understanding how neural networks represent language and other discrete information in continuous vector spaces.  
**Planned topics:**
- Tokenization
- Vocabulary
- Token IDs
- Embeddings
- Embedding spaces
- Positional embeddings
- Language modelling
- Next-token prediction
- Context
- Transformer language models
- Training objectives
- Inference

### 11 — Multimodal & Generative Models
**Status:** Planned  
Expanding beyond text and classification toward modern generative and multimodal systems.  
**Planned topics:**
- Multimodal representations
- Vision-language models
- Image generation
- Diffusion models
- Generative modelling
- Conditioning
- Cross-modal representations
- Audio and other modalities

### 12 — Agentic AI
**Status:** Planned  
Understanding systems that combine models with tools, memory, planning, and interaction.  
**Planned topics:**
- AI agents
- Tool use
- Function calling
- Planning
- Reasoning loops
- Memory
- Retrieval
- Environment interaction
- Agent architectures
- Multi-step execution
- Evaluation of agentic systems

The goal is to understand agentic AI as a system built on top of the underlying machine-learning concepts rather than treating it as a collection of high-level frameworks.

---

## Repository Structure

```text
Deep-Learning-with-PyTorch/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── 01_Tensors/
│   ├── README.md
│   └── main.ipynb
│
├── 02_Autograd/
│   ├── README.md
│   └── main.ipynb
│
├── 03_Neural_Networks/
│   ├── README.md
│   └── main.ipynb
│
├── 04_Classification/
│   ├── README.md
│   └── main.ipynb
│
├── 05_Computer_Vision/
│   ├── README.md
│   └── main.ipynb
│
└── ...
```

**Each chapter contains:**
- `README.md`: Documents the chapter's purpose, concepts, and learning scope.
- `main.ipynb`: Contains the actual experiments, implementations, and observations.

---

## Environment

**Primary development environment:**
- **Operating System:** Ubuntu 26.04 LTS
- **Primary Language:** Python
- **Framework:** PyTorch
- **Notebook Environment:** Jupyter
- **Version Control:** Git + GitHub

**Core dependencies currently include:**
- `torch`
- `numpy`
- `jupyter`
- `matplotlib`

*See `requirements.txt` for the complete project environment.*

---

## Current Understanding

At the current stage, the fundamental PyTorch training pipeline is understood as:

```text
Tensor data
    ↓
Forward computation
    ↓
Prediction
    ↓
Loss
    ↓
Autograd
    ↓
Gradients
    ↓
Parameter update
    ↓
Repeat
```

For classification, this expands to:

```text
Input features / images
        ↓
Linear / convolutional transformations
        ↓
Nonlinear activations
        ↓
Learned representation
        ↓
Class logits
        ↓
CrossEntropyLoss
        ↓
Backpropagation
        ↓
Parameter updates
```

For computer vision, convolution introduces a new way of learning representations:

```text
Image
  ↓
Local patterns
  ↓
Feature maps
  ↓
Higher-level representations
  ↓
Classification
```

The underlying mathematical model remains understood while PyTorch automates the mechanics.

---

## From First Principles to Modern Deep Learning

The learning progression is intentionally structured so that higher-level abstractions are introduced only after their underlying concepts have been understood.

```text
Python
  ↓
NumPy
  ↓
Linear Algebra
  ↓
Calculus
  ↓
Machine Learning
  ↓
Neural Networks
  ↓
PyTorch Tensors
  ↓
Autograd
  ↓
Neural Network Architecture
  ↓
Classification
  ↓
Computer Vision
  ↓
Training & Generalization
  ↓
Sequence Models
  ↓
Attention
  ↓
Transformers
  ↓
Embeddings & Language Models
  ↓
Multimodal & Generative Models
  ↓
Agentic AI
```

This progression is not intended to represent a fixed list of technologies. It represents a progression in understanding.

---

## What This Repository Is Trying to Build

The purpose of this repository is not to collect tutorial code. It is intended to become a record of progressively deeper understanding.

For every major abstraction, the target is:

```text
What problem does this solve?
        ↓
What is happening mathematically?
        ↓
Can I implement the basic mechanism myself?
        ↓
How does PyTorch represent it?
        ↓
What does PyTorch automate?
        ↓
What are the limitations?
```

This approach should make it possible to reason about unfamiliar architectures instead of relying entirely on memorized implementations.

---

## Current Status

🚧 **In Progress**

**Completed**
- ✅ PyTorch Tensors
- ✅ Tensor operations
- ✅ Broadcasting
- ✅ Matrix operations
- ✅ NumPy interoperability
- ✅ Autograd
- ✅ Computational graphs
- ✅ Gradient computation
- ✅ Gradient accumulation
- ✅ `nn.Linear`
- ✅ `MSELoss`
- ✅ SGD
- ✅ Vectorized training
- ✅ Neural Networks
- ✅ Classification fundamentals
- ✅ Cross-entropy classification
- ✅ Nonlinear decision boundaries
- ✅ Two Moons classification
- ✅ Convolution fundamentals
- ✅ CNN architecture
- ✅ CIFAR-10 image classification
- ✅ Pooling
- ✅ Data augmentation
- ✅ CNN training and evaluation
- ✅ Confusion-matrix analysis
- ✅ Visual inspection of CNN predictions

---

## Current Stage

**Next → Training & Generalization**

---

## Long-Term Goal

Build a deep understanding of modern deep learning while retaining the ability to reason about what the framework is doing underneath the abstractions.

The end goal is not merely to:
- Use PyTorch.

It is to understand:
- Why the models work, what the mathematics is doing, what the framework is automating, and where the abstractions come from.

Ultimately, the goal is to progress from understanding individual mathematical operations to being able to reason about complete modern AI systems from first principles.