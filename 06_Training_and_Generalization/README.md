# Training and Generalization

This chapter explores how neural networks learn from training data, how to evaluate performance on unseen data, and how regularization and optimization choices affect learning.

The goal is to understand **why a model generalizes**, how to recognize overfitting, and how training decisions affect learning—not merely to maximize a benchmark score.

## Topics

- Training, validation, and test datasets
- Underfitting, good generalization, and overfitting
- Training loss vs. validation loss
- Building a validation loop in PyTorch
- Evaluating without gradient tracking
- Early stopping and restoring the best checkpoint
- Weight decay
- Dropout and `model.train()` vs. `model.eval()`
- Learning rate and gradient-descent updates
- Comparing learning rates in a controlled experiment
- Learning-rate scheduling with `StepLR`
- Interpreting results without overclaiming

## Dataset and split

The experiments use CIFAR-10, a dataset of small RGB images with 10 classes.

The original CIFAR-10 training partition contains 50,000 images. For development, it is split into:

- 40,000 training images
- 10,000 validation images

The original 10,000-image test partition is kept separate for final evaluation. Avoid repeatedly tuning a model against the test set.

## Core concepts

### Training, validation, and test

- **Training set:** computes gradients and updates model parameters.
- **Validation set:** supports development decisions such as hyperparameter selection, model comparison, and early stopping.
- **Test set:** provides final evaluation after development decisions are complete.

Validation data does not directly update weights, but repeatedly making choices based on validation results can indirectly overfit the development process to that set.

### Underfitting, generalization, and overfitting

| Pattern | Training performance | Validation performance |
|---|---|---|
| Underfitting | Poor | Poor |
| Good generalization | Good | Good |
| Overfitting | Very good | Noticeably worse |

High training accuracy alone does not prove that a model will work well on unseen or real-world data.

### Early stopping

Early stopping stops training after validation performance has failed to improve for a chosen number of epochs (`patience`). The checkpoint with the best monitored validation metric should be restored.

**Best-model epoch and stopping epoch are different:** training can continue after the best epoch while patience is counted.

### Weight decay and dropout

- **Weight decay** regularizes parameter updates by discouraging unnecessarily large weights.
- **Dropout** randomly zeroes selected activations during training. It does not permanently delete neurons.
- `model.train()` activates dropout; `model.eval()` disables it.
- Training accuracy can be lower than validation accuracy when dropout is active during training but disabled during evaluation.

### Learning rate

For a gradient-descent update,

\[
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}
\]

The gradient indicates the local direction of increasing loss; subtracting it moves in the opposite direction. The learning rate η controls the update scale.

- Too small: progress may be slow.
- Suitable: useful progress with stable updates.
- Too large: updates may overshoot, become unstable, or fail to converge.

A learning-rate scheduler changes the learning rate according to a rule. `StepLR`, for example, multiplies it by `gamma` every `step_size` scheduler steps.

## Experiments

### Regularization experiments

Recorded results from individual runs:

| Experiment | Best validation loss | Best validation accuracy |
|---|---:|---:|
| Baseline CNN | 0.9174 | 69.42% |
| CNN + weight decay | 0.9059 | 69.71% |
| CNN + weight decay + dropout | 0.8210 | 71.83% |

These are observed results from individual runs, not universal guarantees. Random initialization, shuffling, dropout, and other sources of randomness can change the outcome.

### Learning-rate experiment

The same initial model weights were used for a three-epoch comparison, with Adam and weight decay held constant.

| Learning rate | Final training loss | Final validation loss |
|---:|---:|---:|
| `0.00001` | 2.0448 | 1.9880 |
| `0.001` | 1.1660 | 1.0834 |
| `0.1` | 2.3160 | 2.3090 |

For this short run, `0.001` performed best among the three tested values. This does not establish it as universally optimal.

## Important distinctions

| Technique | Main role |
|---|---|
| Early stopping | Controls when training ends |
| Weight decay | Regularizes parameter updates |
| Dropout | Randomly removes activations during training |
| Learning-rate scheduling | Changes update scale during training |

## Notebook

Open [`main.ipynb`](./main.ipynb) for explanations, PyTorch implementation, training/validation loops, early stopping, regularization experiments, learning-rate comparison, and a scheduler example.

**Environment note:** The notebook uses `/data/Datasets/Images/First_CNN_CIFAR10_Dataset/` and automatically selects XPU, CUDA, or CPU. Change `DATA_ROOT` if needed.

## Next chapter

**Sequence Models → RNNs → LSTMs/GRUs → Attention → Transformers**
