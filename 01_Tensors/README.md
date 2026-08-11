# Tensors

This chapter introduces the fundamental data structure used throughout PyTorch: the **tensor**.

The goal is to understand tensors not simply as PyTorch objects, but as multidimensional numerical data with well-defined shapes, memory behavior, data types, and devices.

## Topics Covered

* Creating tensors
* Tensor dimensions and axes
* Tensor shape
* Tensor data types
* Tensor devices
* Tensor metadata
* Tensor indexing and slicing
* Tensor views and shared storage
* Tensor cloning
* Tensor reshaping
* Tensor operations
* Element-wise operations
* Broadcasting
* Matrix operations
* NumPy interoperability
* CPU and GPU tensors

## Core Concepts

### Tensor Metadata

For a tensor, the important pieces of metadata include:

* `shape`
* `ndim`
* `dtype`
* `device`

These describe the structure, dimensionality, data type, and hardware device associated with the tensor.

### References and Storage

Python variables can refer to the same tensor object.

Creating a new tensor object is different from creating another reference to an existing tensor.

Understanding object identity and underlying storage is important when working with PyTorch tensors.

### Views

A tensor view can provide a different interpretation of existing storage without copying the underlying data.

Therefore, modifying a view can also modify the original tensor.

### Cloning

`clone()` creates a tensor with independent storage.

This distinction between references, views, and clones becomes important when manipulating tensors safely.

## Shapes and Dimensions

Tensor dimensions describe how data is organized.

For example:

```text
tensor([[1, 2, 3],
        [4, 5, 6]])
```

has:

```text
shape = (2, 3)
```

meaning:

```text
2 rows × 3 columns
```

Understanding shape is essential for neural-network operations and matrix mathematics.

## Broadcasting

Broadcasting allows PyTorch to perform operations between tensors with compatible shapes without explicitly replicating data.

Understanding broadcasting is essential for writing vectorized PyTorch code.

## Learning Philosophy

The chapter focuses on understanding tensor behavior rather than memorizing PyTorch APIs.

The objective is to be able to reason about:

```text
What is the tensor?
What is its shape?
What is its dtype?
Where is it stored?
Does an operation create new storage?
Does it share storage?
What happens to the original tensor?
```

These questions become increasingly important as models and datasets become larger.

## Status

**Completed**

The tensor fundamentals required for the next stage of PyTorch learning have been covered.

