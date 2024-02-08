import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Scalar
scalar = torch.tensor(7)
  # Output:
    # tensor(7)
scalar.ndim
  # Output:
    # 0
# Get tensor back as int
scalar.item()
  # Output:
    # 7

# Vector
vector = torch.tensor([7, 7])
  # Output:
    # tensor([7, 7])
vector.ndim
  # Output:
    # 1
vector.shape
  # Output:
    # torch.Size([2])

# MATRIX
MATRIX = torch.tensor([[7, 8],
                      [9, 10]])
  # Output:
    # tensor([[7, 8],
    #         [9, 10]])
MATRIX.ndim
  # Output:
    # 2
MATRIX[0]
  # Output:
    # tensor([7, 8])
MATRIX.shape
  # Output:
    # torch.Size([2, 2])

# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
  # Output:
    # tensor([[[1, 2, 3],
    #          [3, 6, 9],
    #          [2, 4, 5]]])
TENSOR.ndim
  # Output:
    # 3
TENSOR.shape
  # Output:
    # torch.Size([1, 3, 3])
TENSOR[0]
  # Output:
    #  tensor([[1, 2, 3],
    #          [3, 6, 9],
    #          [2, 4, 5]])

### Random tensors

### Why random Tensors?

### Random tensors are important because the way many neural networks learn is
### that they start with tensors full of random numbers and then adjust those
### random numbers to better represent the data.

### Basic crux of neural networks

### Start with random numbers -> look at data -> update random numbers ->
### look at data -> update random numbers

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)
  # Output:
    # tensor([[0.5727, 0.6081, 0.0619, 0.4617],
    #         [0.4227, 0.4088, 0.2155, 0.0732],
    #         [0.7289, 0.4215, 0.2946, 0.8589]])
random_tensor.ndim
  # Output:
    # 2
# Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(224, 224, 3)) # height, width, color channels (R, G, B)
random_image_size_tensor.shape, random_image_size_tensor.ndim
  # Output:
    # (torch.Size([224, 224, 3]), 3)