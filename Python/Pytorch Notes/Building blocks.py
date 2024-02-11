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

# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
  # Output:
    # tensor([[0., 0., 0., 0.],
    #         [0., 0., 0., 0.],
    #         [0., 0., 0., 0.]])
zeros*random_tensor
  # Output:
    # tensor([[[0., 0., 0., 0.],
    #          [0., 0., 0., 0.],
    #          [0., 0., 0., 0.]],
    #
    #        [[0., 0., 0., 0.],
    #          [0., 0., 0., 0.],
    #          [0., 0., 0., 0.]]])

# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
  # Output:
    # tensor([[1., 1., 1., 1.],
    #         [1., 1., 1., 1.],
    #         [1., 1., 1., 1.]])
ones.dtype
  # Output:
    # torch.float32
random_tensor.dtype
  # Output:
    # torch.float32

# Create a range of tensors and tensors-like
# Use torch.arange()
one_to_ten = torch.arange(start=1, end=11, step=1)
  # Output:
    # tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

# Creating tensors like
ten_zeros = torch.zeros_like(input=one_to_ten)
  # Output:
    # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Tensor datatypes
# Note: Tensor datatypes is one of the 3 big errors you'll run into with Pytorch & deep learning
# 1. Tensors not right datatype
# 2. Tensors not right shape
# 3. Tensors not on the right device


# Float 32 tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # What datatype is the tensor (e.g. float32 or float 16)
                               device=None,# What device is your tensor on
                               requires_grad=False) # Whether or not to track gradients with this tensors operations
  # Output:
    # tensor([3., 6., 9.])
float_32_tensor.dtype
  # Output:
    # torch.float32
float_16_tensor = float_32_tensor.type(torch.float16)
  # Output:
    # tensor([3., 6., 9.], dtype=torch.float16)