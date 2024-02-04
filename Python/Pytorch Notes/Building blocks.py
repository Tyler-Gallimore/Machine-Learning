import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Scalar
  # scalar = torch.tensor(7)
    # Output:
      # tensor(7)
  # scalar.ndim
    # Outputs:
      # 0
  # Get tensor back as int
    # scalar.item()
      # Output:
        # 7

# Vector
  # vector = torch.tensor([7, 7])
    # Output:
      # tensor([7, 7])
  # vector.ndim
    # Output:
      # 1
  # vector.shape
    # Output:
      # torch.Size([2])

# MATRIX
  # MATRIX = torch.tensor([[7, 8],
  #                        [9, 10]])
    # Output:
      # tensor([[7, 8],
      #         [9, 10]])
  # MATRIX.ndim
    # Output:
      # 2
  # MATRIX[0]
    # Output:
      # tensor([7, 8])
  # MATRIX.shape
    # Output:
      # torch.Size([2, 2])

# TENSOR
  # TENSOR = torch.tensor([[[1, 2, 3],
  #                         [3, 6, 9],
  #                         [2, 4, 5]]])
    # Output:
      # tensor([[[1, 2, 3],
      #          [3, 6, 9],
      #          [2, 4, 5]]])
  # TENSOR.ndim
    # Output:
      # 3
  # TENSOR.shape
    # Output:
      # torch.Size([1, 3, 3])
  # TENSOR[0]
    # Output:
      #  tensor([[1, 2, 3],
      #          [3, 6, 9],
      #          [2, 4, 5]])