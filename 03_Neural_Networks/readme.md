# Neural Networks

This chapter builds a neural network from the ground up using PyTorch.

The goal is not simply to use `nn.Module`, `nn.Linear`, and optimizers, but to understand what each component represents and how they work together during training.

## Topics Covered

- Neural network architecture
- `nn.Module`
- `nn.Linear`
- Weights and biases
- Hidden layers and neurons
- ReLU activation
- Why nonlinearities are necessary
- Width and depth
- Learned hidden representations
- Parameter registration
- `model.parameters()`
- Forward propagation
- Loss functions
- Optimizers
- Training loops
- Epochs, batches, and optimization steps
- `TensorDataset`
- `DataLoader`
- Mini-batch training
- Training vs. evaluation mode
- `model.train()`
- `model.eval()`
- `torch.no_grad()`
- Training and validation sets
- Validation loss
- CPU vs. GPU

## Architecture

The main model used in this chapter is:

```text
Input
  │
  ▼
Linear(1 → 16)
  │
  ▼
ReLU
  │
  ▼
Linear(16 → 1)
  │
  ▼
Output
