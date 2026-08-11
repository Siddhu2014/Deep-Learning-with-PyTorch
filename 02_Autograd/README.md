# Autograd

This chapter introduces **PyTorch Autograd**, the automatic differentiation system used to compute gradients.

The chapter begins by building the computational-graph mental model and then connects Autograd to gradient descent and PyTorch's neural-network abstractions.

## Topics Covered

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

## The Computational Graph

When operations involving tensors that require gradients are performed, PyTorch builds a computational graph.

For example:

```text
        x
       / \
     x²   5x
      \   /
        z
```

The graph records the operations required for the backward pass.

The backward pass traverses this graph in reverse and applies the chain rule to calculate gradients.

## `requires_grad`

A tensor created with:

```python
requires_grad=True
```

tells PyTorch to track operations involving that tensor so gradients can later be calculated.

## `grad_fn`

A tensor produced by an operation can contain a `grad_fn`.

`grad_fn` represents the backward operation associated with that tensor.

It is **not the gradient itself**.

`next_functions` can be used to inspect the backward graph.

## Backpropagation

Calling:

```python
loss.backward()
```

starts the backward pass.

Autograd calculates gradients through the computational graph and stores gradients in the `.grad` attributes of the appropriate tensors.

The chain rule is the mathematical foundation of this process.

## Leaf and Non-Leaf Tensors

Leaf tensors are tensors created directly rather than as the result of another tracked operation.

By default, gradients are retained in `.grad` for leaf tensors that require gradients.

Intermediate tensors are non-leaf tensors. Their gradients are not retained by default.

`retain_grad()` can be used when the gradient of a non-leaf tensor needs to be inspected.

## Gradient Accumulation

PyTorch accumulates gradients by default.

Therefore, gradients must be cleared between training iterations.

For an individual tensor:

```python
tensor.grad.zero_()
```

can clear the accumulated gradient.

Optimizers provide their own gradient-reset mechanism.

## `torch.no_grad()`

`torch.no_grad()` temporarily disables gradient tracking.

This prevents PyTorch from building an Autograd graph for operations inside the context.

It is particularly useful when gradients are not required, such as during inference or parameter updates.

## `detach()`

`detach()` creates a new tensor that:

* shares the same underlying storage
* is disconnected from the Autograd graph

It therefore differs from `clone()`.

```text
detach()
    ↓
new tensor object
    ↓
same storage
    ↓
no Autograd connection
```

Whereas:

```text
clone().detach()
    ↓
new tensor object
    ↓
new storage
    ↓
no Autograd connection
```

## Autograd and Gradient Descent

Autograd can replace manually derived gradients in a training loop.

The fundamental training cycle is:

```text
Forward pass
    ↓
Prediction
    ↓
Loss
    ↓
backward()
    ↓
Gradients
    ↓
Parameter update
    ↓
Clear gradients
    ↓
Repeat
```

For a simple neuron:

$$
\hat{y}=wx+b
$$

the trainable parameters are `w` and `b`.

Autograd calculates their gradients automatically.

## Vectorized Training

The same neuron can process multiple training examples simultaneously using tensors.

For example:

```text
x₁ → y₁
x₂ → y₂
x₃ → y₃
x₄ → y₄
x₅ → y₅
```

The same parameters are used across all examples, and the loss combines information from the complete dataset.

This connects Autograd with the tensor and vectorization concepts introduced in the previous chapter.

## PyTorch Neural-Network Abstractions

After understanding the underlying mechanics, the chapter introduces PyTorch's higher-level abstractions.

### `nn.Linear`

Instead of manually creating trainable weights and biases:

```python
w = torch.tensor(..., requires_grad=True)
b = torch.tensor(..., requires_grad=True)
```

PyTorch provides:

```python
nn.Linear(1, 1)
```

which represents a trainable linear transformation.

The underlying mathematics remains:

$$
y=xW+b
$$

### Loss Functions

`nn.MSELoss()` provides the Mean Squared Error loss used in the examples.

### Optimizers

`torch.optim.SGD` automates the parameter-update step.

The resulting training loop becomes:

```text
forward
   ↓
loss
   ↓
backward
   ↓
optimizer.step()
   ↓
optimizer.zero_grad()
   ↓
repeat
```

The mathematics has not changed. PyTorch is simply automating the mechanics.

## Learning Rate

The learning rate controls the size of each parameter update.

A learning rate that is too large can cause the parameters to overshoot the minimum and oscillate.

A smaller learning rate can produce smoother convergence, although convergence may take more iterations.

There is no universally correct learning rate; it depends on the optimization problem.

## Key Understanding

The important transition in this chapter is:

```text
Manual mathematics
       ↓
Manual gradients
       ↓
Manual parameter updates
       ↓
Autograd
       ↓
PyTorch abstractions
```

The framework does not replace the underlying mathematics.

It automates the repetitive mechanics while the model designer still controls the architecture, loss function, optimizer, learning rate, data, and training procedure.

## Status

**Completed**

The fundamentals of Autograd, gradient-based optimization, and the basic PyTorch training workflow have been covered.

### Next

The next chapter moves from a single linear neuron to actual **neural networks**, including multiple layers and nonlinear activation functions.

