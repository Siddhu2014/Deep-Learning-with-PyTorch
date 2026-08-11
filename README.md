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

For example, before using Autograd, the underlying ideas of derivatives, gradients, the chain rule, and gradient descent were explored manually.

Similarly, before using `nn.Linear`, a trainable neuron was implemented directly using tensors, weights, bias, loss, gradients, and parameter updates.

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

The chapter establishes the tensor mental model required for everything that follows.

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

The chapter connects the mathematical concept of backpropagation with PyTorch's automatic differentiation engine.

[Open `02_Autograd/main.ipynb`](02_Autograd/main.ipynb)

---

## 03 — Neural Networks

**Status: Next**

Moving from a single linear neuron to actual neural networks.

Planned topics:

* `nn.Module`
* Multiple layers
* Parameters
* `nn.Sequential`
* Activation functions
* Forward propagation
* Multi-layer networks
* Nonlinear transformations
* Network architecture

---

## 04 — Training

**Status: Planned**

Going deeper into the process of training neural networks.

Planned topics:

* Loss functions
* Optimizers
* Training loops
* Learning rates
* Validation
* Evaluation
* Overfitting
* Underfitting
* Regularization
* Training diagnostics

---

## 05 — Computer Vision

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

## 06 — Sequence Models

**Status: Planned**

Understanding models designed for sequential data.

Planned topics:

* Sequential data
* RNNs
* LSTMs
* Hidden states
* Sequence modelling

---

## 07 — Transformers

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

A simple trainable neuron can be represented as:

$$
\hat{y}=wx+b
$$

and trained using gradient descent.

PyTorch now automates the mechanics:

```text
nn.Linear
    ↓
Loss function
    ↓
Autograd
    ↓
Optimizer
    ↓
Updated parameters
```

while the underlying mathematical model remains understood.

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

Current stage:

**Next → Multi-layer Neural Networks**

---

## Long-Term Goal

Build a deep understanding of modern deep learning while retaining the ability to reason about what the framework is doing underneath the abstractions.

The end goal is not merely to **use PyTorch**.

It is to understand **why it works**.

