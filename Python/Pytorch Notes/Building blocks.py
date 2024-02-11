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

### Tensor datatypes
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
float_16_tensor * float_32_tensor
  # Output:
    # tensor([ 9., 36., 81.])
int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
  # Output:
    # tensor([3, 6, 9], dtype=torch.int32)

### Getting information from tensors (tensor attributes)
# 1. Tensors not right datatype - to get datatype from a tensor, can use 'tensor.dtype'
# 2. Tensors not right shape - to get shape from a tensor, can use 'tensor.shape'
# 3. Tensors not on the right device - to get device from a tensor, can use 'tensor.device'

# Create a tensor
some_tensor = torch.rand(3, 4,)
  # Output:
    # tensor([[0.6396, 0.5449, 0.9678, 0.5607],
    #         [0.1891, 0.3000, 0.3390, 0.4726],
    #         [0.4787, 0.4789, 0.7077, 0.5334]])

# Find out details about some tensor
print(some_tensor)
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device tensor is on: {some_tensor.device}")
  # Output:
    # tensor([[0.1437, 0.2829, 0.6995, 0.3229],
    #    [0.9369, 0.4691, 0.1165, 0.3444],
    #    [0.2558, 0.4425, 0.1066, 0.3839]])
    # Datatype of tensor: torch.float32
    # Shape of tensor: torch.Size([3, 4])
    # Device tensor is on: cpu

### Manipulating Tensors (tensor operations)

# Tensor operations include:
# Addition
# Subtraction
# Multiplication (element-wise)
# Division
# Matrix multiplication

# Create a tensor and add 10 to it
tensor = torch.tensor([1, 2 ,3])
tensor + 10
  # Output:
    # tensor([11, 12, 13])

# Multiply tensor by 10
tensor * 10
  # Output:
    # tensor([10, 20, 30])

# Subtract 10
tensor - 10
  # Output:
    # tensor([-9, -8, -7])

# Try out Pytorch in-built functions
torch.mul(tensor, 10)
  # Output:
    # tensor([10, 20, 30])

### Matrix multiplication
# Two main ways of performing multiplcation in neural networks and deep learning:

# 1. Element-wise multiplication
# 2. Matrix multiplication (dot product)

# There are two main rules that performing matrix multiplication needs to satisfy:
#  1. The **inner dimensions** must match:
#    '(3, 2) @ (3, 2)' won't work
#    '(2, 3) @ (3, 2)' will work
#    '(3, 2) @ (2, 3)' will work
#  2. The resulting matrix has the shape of the **outer dimenstions**:
#    '(2, 3) @ (3, 2)' -> '(2, 2)'
#    '(3, 2) @ (2, 3)' -> '(3, 3)'

# Element wise multiplication
print(tensor, "*", tensor)
print(f"Equals: {tensor * tensor}")
  # Output:
    # tensor([1, 2, 3]) * tensor([1, 2, 3])
    # Equals: tensor([1, 4, 9])

# Matrix multiplication
torch.matmul(tensor, tensor)
  # Output:
    # tensor(14)

# Matrix multiplication by hand
# 1*1 + 2*2 + 3*3
  # Output:
    # 14

# Time difference for using loop to matmul
%%time
value = 0
for i in range(len(tensor)):
  value += tensor[i] * tensor[i]
print(value)
  # Output:
    # tensor(14)
    # CPU times: user 1.46 ms, sys: 0 ns, total: 1.46 ms
    # Wall time: 1.47 ms

%%time
torch.matmul(tensor, tensor)
  # Output:
    # CPU times: user 100 µs, sys: 8 µs, total: 108 µs
    # Wall time: 114 µs
    # tensor(14)