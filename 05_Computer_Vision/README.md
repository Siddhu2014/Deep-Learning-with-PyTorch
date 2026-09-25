# Computer Vision — Convolutional Neural Networks

This chapter covers **Convolutional Neural Networks (CNNs)** with PyTorch using CIFAR-10.

The goal is to understand the CNN workflow rather than spend excessive time optimizing a small benchmark.

## Topics

- `Conv2d`
- Filters, channels, and feature maps
- Kernel size, stride, and padding
- ReLU activation
- Max pooling
- Flattening convolutional features
- Fully connected classification
- `CrossEntropyLoss`
- Adam optimization
- Complete epoch-based training loops
- PyTorch XPU acceleration
- Training vs. test accuracy
- Confusion matrices
- Model capacity and generalization
- Data augmentation
- Visual inspection of misclassified images

## Dataset

CIFAR-10 contains:

- 50,000 training images
- 10,000 test images
- RGB images
- `32 × 32` resolution
- 10 classes

| Index | Class |
|---:|---|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

## Final CNN

```text
Input: 3 × 32 × 32
        ↓
Conv2d: 3 → 32, 3×3, padding=1
        ↓
ReLU
        ↓
MaxPool 2×2
        ↓
Conv2d: 32 → 64, 3×3, padding=1
        ↓
ReLU
        ↓
MaxPool 2×2
        ↓
Flatten: 64 × 8 × 8 = 4096
        ↓
Linear: 4096 → 10
```

The two pooling layers reduce:

```text
32 × 32 → 16 × 16 → 8 × 8
```

Therefore the flattened representation is:

```text
64 × 8 × 8 = 4096
```

## Training configuration

- Optimizer: Adam
- Learning rate: `0.001`
- Batch size: `64`
- Main experiment: `20` epochs
- Training augmentation:
  - `RandomHorizontalFlip`
  - `RandomCrop(32, padding=4)`
- Test transform:
  - `ToTensor()`

## Experiments

| Experiment | Training accuracy | Test accuracy |
|---|---:|---:|
| 16 → 32 channels, 10 epochs | 70.72% | 66.70% |
| 16 → 32 channels, 20 epochs | 73.74% | 68.63% |
| 32 → 64 channels, 20 epochs | 81.92% | 70.27% |
| 32 → 64 + horizontal flip | 76.98% | 72.24% |
| 32 → 64 + flip + random crop | 71.05% | **73.80%** |

### What the experiments showed

Increasing model capacity substantially improved training performance, while the improvement on unseen data was smaller.

Data augmentation reduced training accuracy while improving test accuracy. This demonstrated the distinction between **fitting** and **generalization**.

The final measured test accuracy was **73.80%**.

## Important dimension formula

For a convolution:

\[
\text{Output size} = \frac{N-K+2P}{S}+1
\]

where:

- `N` = input spatial size
- `K` = kernel size
- `P` = padding
- `S` = stride

The positional reasoning behind this formula is more important than memorizing it.

## Notebook

Open [`main.ipynb`](./main.ipynb) for the complete implementation and experiments.

## Next chapter

**Sequence Models → RNNs → LSTMs/GRUs → Attention → Transformers**
