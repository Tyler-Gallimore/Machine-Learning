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
# %%time
value = 0
for i in range(len(tensor)):
  value += tensor[i] * tensor[i]
print(value)
  # Output:
    # tensor(14)
    # CPU times: user 1.46 ms, sys: 0 ns, total: 1.46 ms
    # Wall time: 1.47 ms

# %%time
torch.matmul(tensor, tensor)
  # Output:
    # CPU times: user 100 µs, sys: 8 µs, total: 108 µs
    # Wall time: 114 µs
    # tensor(14)

### One of the most common erros in deep learning: shape errors
# Shapes for matrix multiplication
tensor_a = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])

tensor_b = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])

torch.mm(tensor_a, tensor_b) # torch.mm is the same as torch.matmul (it's an alias for writing less code)
  # Output:
    # ---------------------------------------------------------------------------
    # RuntimeError                              Traceback (most recent call last)
    # <ipython-input-6-eddc2caa8b2b> in <cell line: 11>()
    #       9                          [9, 12]])
    #      10 
    # ---> 11 torch.mm(tensor_a, tensor_b) # torch.mm is the same as torch.matmul (it's an alias for writing less code)

    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)

# To fix our tensor shape issues, we can manipulate the shape of one of our tensors using a **transpose**
# A **transpose** swithes the axes or dimensions of a given tensor 
tensor_b, tensor_b.shape
  # Output:
    # (tensor([[ 7, 10],
    #          [ 8, 11],
    #          [ 9, 12]]),
    # torch.Size([3, 2]))

tensor_b.T, tensor_b.T.shape
  # Output:
    # (tensor([[ 7,  8,  9],
    #          [10, 11, 12]]),
    # torch.Size([2, 3]))

# The matrix multiplication operation when tensor_b is transposed
print(f"Original shapes: tensor_a = {tensor_a.shape}, tensor_b = {tensor_b.shape}")
print(f"New shapes: tensor_a = {tensor_a.shape} (same shape as above),  tensor_b.T = {tensor_b.T.shape}")
print(f"Multiplying: {tensor_a.shape} @ {tensor_b.T.shape} <- innter dimensions must match")
print("Output:\n")
output = torch.matmul(tensor_a, tensor_b.T)
print(output)
print(f"\nOutput shape: {output.shape}")
  # Output:
    # Original shapes: tensor_a = torch.Size([3, 2]), tensor_b = torch.Size([3, 2])
    # New shapes: tensor_a = torch.Size([3, 2]) (same shape as above),  tensor_b.T = torch.Size([2, 3])
    # Multiplying: torch.Size([3, 2]) @ torch.Size([2, 3]) <- innter dimensions must match
    # Output:

    # tensor([[ 27,  30,  33],
    #         [ 61,  68,  75],
    #         [ 95, 106, 117]])

    # Output shape: torch.Size([3, 3])

### Finding the min, max, mean, sum, etc (tensor aggregation)
# Create a tensor
x = torch.arange(1, 100, 10)

# Find the min
torch.min(x), x.min()
  # Output:
  #  (tensor(0), tensor(0))

# Find the max
torch.max(x), x.max()
  # Output:
    # (tensor(90), tensor(90))

# Find the mean - note: the torch.mean() function requires a tensor of float32 datatype to work
torch.mean(x.type(torch.float32)), x.type(torch.float32).mean()
  # Output:
    # (tensor(45.), tensor(45.))

# Find the sum
torch.sum(x), x.sum()
  # Output:
    # (tensor(450), tensor(450))

### Finding the positional min and max

# Find the position in tensor that has the minimum value with argmin() -> returns index postion of target tensor where the minimum value occurs
x.argmin()
  # Output:
    # tensor(0)

# Find the postion in tensor that has the maxium value with argmax()
x.argmax()
  # Output:
    # tensor(9)

## Reshaping, stacking, squeezing and unsqueezing tensors

# Reshaping - reshapes an input tensor to a defined shape
# View - Return a view of an input tensor of certain shape but keep the same memory as the original tensor
# Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
# Squeeze - removes all '1' dimensions from a tensor
# Unsqueeze - adds a '1' dimension to a target tensor
# Permute - Return a view of the input with dimensions permuted (swapped) in a certain way

# Let's create a tensor
import torch
x = torch.arange(1, 10)
x, x.shape
  # Output:
    # (tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]), torch.Size([9]))

# Add an extra dimension
x_reshaped = x.reshape(1, 9)
x_reshaped, x_reshaped.shape
  # Output:
    # (tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]]), torch.Size([1, 9]))

# Change the view
z = x.view(1, 9)
z, z.shape
  # Output:
    # (tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]]), torch.Size([1, 9]))

# Changing z changes x (because a view of a tensor shares the same memory as the original input)
z[:, 0] = 5
z, x
  # Output:
    # (tensor([[5, 2, 3, 4, 5, 6, 7, 8, 9]]), tensor([5, 2, 3, 4, 5, 6, 7, 8, 9]))

# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0)
x_stacked
  # Output:
    # tensor([[5, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [5, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [5, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [5, 2, 3, 4, 5, 6, 7, 8, 9]])

# torch.squeeze() - removes all single dimensions from a target tensor
print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimensions from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")
  # Output:
    # Previous tensor: tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    # Previous shape: torch.Size([1, 9])

    # New tensor: tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # New shape: torch.Size([9])

# torch.unsqueeze() - adds a single dimension to a target tensor at a specific dim (dimension)
print(f"Previous target: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")
  # Output:
    # Previous target: tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # Previous shape: torch.Size([9])

    # New tensor: tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    # New shape: torch.Size([1, 9])

# torch.permute - rearranges the dimension of a target tensor in a specified order
x_original = torch.rand(size=(224, 224, 3)) # [height, width, color_channels]

# Permute the original tensor to rearragne the axis (or dim) order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}") # [color_channels, height, width]
  # Output:
    # Previous shape: torch.Size([224, 224, 3])
    # New shape: torch.Size([3, 224, 224])

## Indexing (selecting data from tensors)

# Indexing with Pytorch is similar to indexing with NumPy

# Create a tensor
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape
  # Output:
    # (tensor([[[1, 2, 3],
    #           [4, 5, 6],
    #           [7, 8, 9]]]),
    # torch.Size([1, 3, 3]))

# Let's index on our new tensor
x[0]
  # Output:
    # tensor([[1, 2, 3],
    #         [4, 5, 6],
    #         [7, 8, 9]])

# Let's index on the middle bracket (dim=1)
x[0, 0]
  # Output:
    # tensor([1, 2, 3])

# Let's index on the most inner bracket (last dimension)
x[0, 2, 2]
  # Output:
    # tensor(9)

# You can also use ":" to select "all" of a target dimension
x[:, 0]
  # Output:
    # tensor([[1, 2, 3]])

# Get all values of 0th and 1st dimensions but only index 1 of 2nd dimension
x[:, :, 1]
  # Output:
    # tensor([[2, 5, 8]])

# Get all values of the 0 dimention but only the 1 index value of the 1st and 2nd dimension
x[:, 1, 1]
  # Output:
    # tensor([5])

# Get index 0 of 0th and 1st dimension and all values of 2nd dimension
x[0, 0, :]
  # Output:
    # tensor([1, 2, 3])

# Index on x to return 9
x[0, 2, 2]

# Index on x to return 3, 6, 9
x[0, :, 2:3]
  # Output:
    # tensor([[3],
    #         [6],
    #         [9]])

# Pytorch tensors & NumPy

# NumPy is a popular scientific Python numerical computing library
# And because of this, Pytorch has functionality to interact with it.

# Data in NumPy, want in Pytorch tensor -> 'torch.from_numpy(ndarray)'
# Pytorch tensor -> NumPy -> torch.Tensor.numpy()

# NumPy array to tensor
import torch
import numpy as np

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) # warning: when converting from numpy -> pytorch, pytorch reflects numpy's default datatype of float64 unless specified otherwise.
array, tensor
  # Output:
    # (array([1., 2., 3., 4., 5., 6., 7.]),
    #  tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64))

# Change the value of array, what will this do to 'tensor'?
array = array + 1
array, tensor
  # Output:
    # (array([2., 3., 4., 5., 6., 7., 8.]),
    #  tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64))

# Tensor to NumPy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
tensor, numpy_tensor
  # Output:
    # (tensor([1., 1., 1., 1., 1., 1., 1.]),
    #  array([1., 1., 1., 1., 1., 1., 1.], dtype=float32))

# Change the tensor, what happens to 'numpy_tensor'?
tensor = tensor + 1
tensor, numpy_tensor
  # Output:
    # (tensor([2., 2., 2., 2., 2., 2., 2.]),
    #  array([1., 1., 1., 1., 1., 1., 1.], dtype=float32))

## Reproducbility (trying to take the random out of random)

# In short how a neural network learns:
# 'start with random numbers -> tensor operations -> update random numbers to try and make them better representations of the data -> again -> again -> again...'

# To reduce the randomness in neural networks and Pytorch comes a concept of a **random seed**.
# Essentially what the random seed does is "flavour" the randomness.

import torch

# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)
  # Output:
    # tensor([[0.4016, 0.5288, 0.4627, 0.5267],
    #         [0.3326, 0.3860, 0.0616, 0.3697],
    #         [0.5452, 0.2601, 0.5551, 0.5930]])
    # tensor([[0.9441, 0.2256, 0.0439, 0.2187],
    #         [0.4927, 0.5350, 0.5622, 0.4081],
    #         [0.4948, 0.7445, 0.1453, 0.1287]])
    # tensor([[False, False, False, False],
    #         [False, False, False, False],
    #         [False, False, False, False]])

# Let's make some random but reproducible tensors
import torch

# Set the random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)
  # Output:
    # tensor([[0.8823, 0.9150, 0.3829, 0.9593],
    #         [0.3904, 0.6009, 0.2566, 0.7936],
    #         [0.9408, 0.1332, 0.9346, 0.5936]])
    # tensor([[0.8823, 0.9150, 0.3829, 0.9593],
    #         [0.3904, 0.6009, 0.2566, 0.7936],
    #         [0.9408, 0.1332, 0.9346, 0.5936]])
    # tensor([[True, True, True, True],
    #         [True, True, True, True],
    #         [True, True, True, True]])

## Running tensors and Pytorch objects on the GPUs (and making faster computations)

# GPUS = faster computation on numbers, thanks to CUDA + NVIDIA hardware + PyTorch working behind the scenes to make everything hunky dory (good).

### 1. Getting a GPU

# 1. Easiest - Use Google Colab for a free GPU. (options to upgrade as well)
# 2. Use your own GPU - takes a little bit of setup and requires the investment of puchasing a GPU, there's lots of options..
# 3. Use cloud computing - GCP, AWS, Azure, these services allow you to rent computers on the cloud and access them

# For 2, 3 PyTorch + GPU drivers (CUDA) takes a little bit of setting up, to do this, refer to PyTorch setup documentation

### 2. Check for GPU access with PyTorch
import torch
torch.cuda.is_available()
  # Output:
    # True

# For PyTorch since it's capable of running compute on the GPU or CPU, it's best practice to setup device agnostic code

# E.g. run on GPU if available, else default to CPU

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
  # Output:
    # 'cuda'

# Count number of devices
torch.cuda.device_count()
  # Output:
    # 1

## 3. Putting tensors (and models) on the GPU

# The reason we want our tensors/models on the GPU is because using a GPU results in faster computations.

# Create a tensor (default on the CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)
  # Output:
    # tensor([1, 2, 3]) cpu

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
tensor_on_gpu
  # Output:
    # tensor([1, 2, 3], device='cuda:0')

### 4. Moving tensors back to the CPU
# If tensor is on GPU, can't transform it to NumPy
tensor_on_gpu.numpy()
  # Output:
    # ---------------------------------------------------------------------------
    # TypeError                                 Traceback (most recent call last)
    # <ipython-input-10-f273530b6b64> in <cell line: 3>()
    #       1 ### 4. Moving tensors back to the CPU
    #       2 # If tensor is on GPU, can't transform it to NumPy
    # ----> 3 tensor_on_gpu.numpy()

    # TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

# To fix the GPU tensor with NumPy issue, we can first set it to the CPU
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
tensor_back_on_cpu
  # Output:
    # array([1, 2, 3])