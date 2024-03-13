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