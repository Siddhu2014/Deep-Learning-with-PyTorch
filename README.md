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
```

The objective is to develop the ability to **reason about PyTorch code**, not simply memorize APIs.

---

# Learning Path

## 01 — Tensors

**Status: Completed**

The foundation of PyTorch.

Topics covered:

* Tensor creation
* Tensor dimensions and axes
* Tensor shapes
* Tensor data types
* Tensor devices
* Tensor metadata
* Tensor indexing and slicing
* Tensor views
* Shared storage
* Tensor cloning
* Reshaping
* Tensor operations
* Element-wise operations
* Broadcasting
* Matrix operations
* NumPy interoperability
* CPU and GPU tensors

[Open `01_Tensors/main.ipynb`](01_Tensors/main.ipynb)

---

## 02 — Autograd

**Status: Completed**

Understanding how PyTorch computes gradients automatically.

Topics covered:

* `requires_grad`
* Computational graphs
* `grad_fn`
* `next_functions`
* Branching computational graphs
* `backward()`
* Leaf and non-leaf tensors
* `retain_grad()`
* Gradient accumulation
* `torch.no_grad()`
* `requires_grad_(False)`
* `detach()`
* `clone()` vs `detach()`
* Trainable parameters
* Gradient descent
* Weight and bias
* Vectorized training
* `nn.Linear`
* `MSELoss`
* SGD
* Complete PyTorch training loops

[Open `02_Autograd/main.ipynb`](02_Autograd/main.ipynb)

---

## 03 — Neural Networks

**Status: Completed**

Moving from a single linear neuron to multi-layer neural networks.

Topics covered:

* `nn.Module`
* Multiple layers
* Parameters
* Activation functions
* Forward propagation
* Multi-layer networks
* Nonlinear transformations
* Network width and depth
* Training and validation
* `DataLoader`
* CPU vs GPU for small workloads

[Open `03_Neural_Networks/main.ipynb`](03_Neural_Networks/main.ipynb)

---

## 04 — Classification

**Status: Completed**

Understanding classification from logits through nonlinear decision boundaries.

Topics covered:

* Classification targets and class indices
* Multi-class classifiers
* `CrossEntropyLoss`
* Logits
* `argmax` and class prediction
* Training and validation accuracy
* `model.train()` and `model.eval()`
* `torch.no_grad()`
* Model width and depth experiments
* Run-to-run variation
* Decision-boundary visualization
* Linear vs nonlinear classification
* ReLU and nonlinear decision boundaries
* Two Moons dataset

[Open `04_Classification/main.ipynb`](04_Classification/main.ipynb)

---

## 05 — Training

**Status: Planned**

Going deeper into the process of training neural networks.

Planned topics:

* Loss functions
* Optimizers
* Learning rates
* Validation
* Evaluation
* Overfitting
* Underfitting
* Regularization
* Training diagnostics

---

## 06 — Computer Vision

**Status: Planned**

Applying deep learning to visual data.

Planned topics:

* Image tensors
* Convolution
* Convolutional Neural Networks
* Pooling
* Image classification
* Training CNNs
* Transfer learning

---

## 07 — Sequence Models

**Status: Planned**

Understanding models designed for sequential data.

Planned topics:

* Sequential data
* RNNs
* LSTMs
* Hidden states
* Sequence modelling

---

## 08 — Transformers

**Status: Planned**

Understanding the architecture behind modern language and multimodal models.

Planned topics:

* Attention
* Self-attention
* Query, Key, Value
* Positional information
* Multi-head attention
* Transformer architecture

---

# Repository Structure

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
└── ...
```

Each chapter contains:

```text
README.md
main.ipynb
```

The README documents the chapter's purpose, concepts, and learning scope.

The notebook contains the actual experiments and implementations.

---

# Environment

Primary development environment:

* **Operating System:** Ubuntu 26.04 LTS
* **Primary Language:** Python
* **Framework:** PyTorch
* **Notebook Environment:** Jupyter
* **Version Control:** Git + GitHub

Core dependencies currently include:

```text
torch
numpy
jupyter
matplotlib
```

See [`requirements.txt`](requirements.txt) for the project environment.

---

# Current Understanding

At the current stage, the fundamental PyTorch pipeline is understood as:

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
Input features
    ↓
Linear transformations
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

The underlying mathematical model remains understood while PyTorch automates the mechanics.

---

# Guiding Principle

The goal of this repository is **not** to become a collection of copied PyTorch tutorials.

It is a record of building deep-learning understanding from the foundations upward.

The progression is intentionally:

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
Deep Learning
```

Every abstraction should eventually be understandable in terms of the concepts underneath it.

---

# Status

🚧 **In Progress**

Completed:

* ✅ Tensors
* ✅ Autograd
* ✅ Basic gradient-based training
* ✅ `nn.Linear`
* ✅ `MSELoss`
* ✅ SGD
* ✅ Vectorized training
* ✅ Neural Networks
* ✅ Classification fundamentals
* ✅ Nonlinear decision boundaries

Current stage:

**Next → Deeper Training Concepts**

---

## Long-Term Goal

Build a deep understanding of modern deep learning while retaining the ability to reason about what the framework is doing underneath the abstractions.

The end goal is not merely to **use PyTorch**.

It is to understand **why it works**.
