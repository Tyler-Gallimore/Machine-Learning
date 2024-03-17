# 02. Neural Network classification with PyTorch

# Classification is a problem of predicting whether something is one thing or another (there can be multiple things are the options)
## 1. Make classification data and get it ready
import sklearn
from sklearn.datasets import make_circles

# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

len(X), len(y)
  # Output:
    # (1000, 1000)

print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")
  # Output:
    # First 5 samples of X:
    # [[ 0.75424625  0.23148074]
    # [-0.75615888  0.15325888]
    # [-0.81539193  0.17328203]
    # [-0.39373073  0.69288277]
    # [ 0.44220765 -0.89672343]]
    # First 5 samples of y:
    # [1 1 1 1 0]

# Make DataFrame of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0], 
                        "X2": X[:, 1],
                        "label": y})
circles.head(10)
  # Output:
    # 	  X1          X2 label
#0	 0.754246	0.231481	1
#1	-0.756159	0.153259	1
#2	-0.815392	0.173282	1
#3	-0.393731	0.692883	1
#4	 0.442208  -0.896723	0
#5	-0.479646	0.676435	1
#6	-0.013648	0.803349	1
#7	 0.771513	0.147760	1
#8	-0.169322  -0.793456	1
#9	-0.121486	1.021509	0

# Visualize, visualize, visualize
import matplotlib.pyplot as plt
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu);
  # Output:
    # Circle chart with two circles, Red on outside and Blue inside

# Note: The data we're working with if often referred to as a toy dataset, a dataset that is small enough to experiment but still sizeable enough to preactice the fundamentals.

### 1.1 Check input and output shapes

X.shape, y.shape
  # Output:
    # ((1000, 2), (1000,))

X
  # Output:
    # array([[ 0.75424625,  0.23148074],
    #        [-0.75615888,  0.15325888],
    #        [-0.81539193,  0.17328203],
    #        ...,
    #        [-0.13690036, -0.81001183],
    #        [ 0.67036156, -0.76750154],
    #        [ 0.28105665,  0.96382443]])

# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]

print(f"Values for one sample of X: {X_sample} and the sample of y: {y_sample}")
print(f"shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")
  # Output:
    # Values for one sample of X: [0.75424625 0.23148074] and the sample of y: 1
    # shapes for one sample of X: (2,) and the same for y: ()

import torch
torch.__version__
  # Output:
    # 2.2.1+cu121

type(X)
  # Output:
    # numpy.ndarray

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X[:5], y[:5]
  # Output:
    # (tensor([[ 0.7542,  0.2315],
    #          [-0.7562,  0.1533],
    #          [-0.8154,  0.1733],
    #          [-0.3937,  0.6929],
    #          [ 0.4422, -0.8967]]),
    #  tensor([1., 1., 1., 1., 0.]))

type(X), X.dtype, y.dtype
  # Output:
    # (torch.Tensor, torch.float32, torch.float32)

# Split data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 0.2 = 20% of data will be test & 80% will be train
                                                    random_state=42)

len(X_train), len(X_test), len(y_train), len(y_test)
  # Output:
    # (800, 200, 800, 200)

## 2. Building a model

# Let's build a model to classify our blue and red dots.

# To do so, we want to:
# 1. Setup device agnostic code so our code will run on an accelerator (GPU) if there is one
# 2. Construct a model (by sublcassing 'nn.Module')
# 3. Define a loss function and optimizer
# 4. Create a training and test loop

# Import PyTorch and nn
import torch
from torch import nn

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Now we've setup device agnostic code, let's create a model that:

# 1. Subclass 'nn.module' (almost all models in PyTorch sublcass 'nn.module')
# 2. Create 2 'nn.Linear()' layers that are capable of handling the shapes of our data
# 3. Defines a 'forward()' method that outlines the forward pass (or forward computation) of the model
# 4. Instatiate an instance of our model class and send it to the target device

from sklearn import datasets
# 1. Construct a model that sublcasses nn.module
class CircleModelV0(nn.Module):
  def __init__(self):
    super().__init__()
    # 2. Create 2 nn.Linear layers capable of handling the shapes of our data
    self.layer_1 = nn.Linear(in_features=2, out_features=5) # Takes in 2 features and upscales to 5 features
    self.layer_2 = nn.Linear(in_features=5, out_features=1) # Takes in 5 features from previous layer and outputs a single feature (same shape as y)
  
  # 3. Define a forward() method that outlines the forward pass
  def forward(self, x):
    return self.layer_2(self.layer_1(x)) # x -> layer_1 -> layer_2 -> output

# 4. Instantiate an instance of our model class and send it to the target device
model_0 = CircleModelV0().to(device)
model_0
  # Output:
    # CircleModelV0(
    #  (layer_1): Linear(in_features=2, out_features=5, bias=True)
    #  (layer_2): Linear(in_features=5, out_features=1, bias=True)
    # )
