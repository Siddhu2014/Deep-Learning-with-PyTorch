# Classification

This chapter builds classifiers with PyTorch and focuses on understanding what changes when the task moves from regression to classification.

## Topics Covered

- Classification targets and class indices
- `nn.Module`
- Multi-layer classifiers
- Logits
- `argmax` and class prediction
- `CrossEntropyLoss`
- Training and validation splits
- `DataLoader` and mini-batch training
- Training accuracy and validation accuracy
- `model.train()` and `model.eval()`
- `torch.no_grad()`
- Model width and depth
- Random initialization and run-to-run variation
- Decision-boundary visualization
- Linear vs. nonlinear classification
- ReLU and nonlinear decision boundaries

## Main Architecture

The basic classifier used in the chapter is:

```text
Input: 2 features
      ↓
Linear(2 → 16)
      ↓
ReLU
      ↓
Linear(16 → 3)
      ↓
3 logits
```

For classification, the final layer produces **logits**, not probabilities. `CrossEntropyLoss` consumes the raw logits directly, so Softmax is not added to the model during training.

## Experiments

The chapter deliberately changes the difficulty of the data and compares model architectures.

Repeated experiments on the harder three-class dataset produced these mean validation accuracies:

| Architecture | Mean validation accuracy |
|---|---:|
| `2 → 16 → 3` | 76.00% |
| `2 → 32 → 3` | 82.00% |
| `2 → 64 → 3` | 80.67% |
| `2 → 16 → 16 → 3` | 79.33% |
| `2 → 32 → 32 → 3` | 81.33% |

These are toy experiments, not universal benchmarks. They demonstrate that increasing capacity can help, but more neurons or more layers do **not** automatically produce better validation performance.

## Nonlinear Classification

The chapter then uses the Two Moons dataset to demonstrate why nonlinear activations matter.

A linear classifier:

```text
2 → 2
```

can only learn a linear decision boundary.

A network such as:

```text
2 → 32 → 32 → 2
      ↓       ↓
     ReLU    ReLU
```

can learn nonlinear decision boundaries and separate the curved classes much more effectively.

## Decision Boundaries

Because the experiments use two input features, the notebook visualizes the model's prediction across a 2D grid. This makes the learned decision regions directly visible.

## Key Mental Model

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

## Status

✅ **Classification fundamentals complete**

Next: deeper training concepts and model evaluation.
