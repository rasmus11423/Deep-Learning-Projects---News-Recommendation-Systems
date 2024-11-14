import numpy as np 

print(np.max([1,2]))

import torch

# Check PyTorch version
print(torch.__version__)

# Test Tensor creation
x = torch.rand(3, 3)
print(x)


import polars as pl

# Test functionality
df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35]
})

print(df)

import tensorflow as tf

# Check TensorFlow version
print("TensorFlow Version:", tf.__version__)

# Test basic computation
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = tf.add(a, b)
print("TensorFlow Computation:", c)


from tqdm import tqdm
import time

# Test tqdm progress bar
for i in tqdm(range(10), desc="Processing"):
    time.sleep(0.1)  # Simulate work

import yaml

# Example YAML string
yaml_str = """
name: John Doe
age: 30
skills:
  - Python
  - Machine Learning
"""

# Load YAML string as a dictionary
data = yaml.safe_load(yaml_str)
print(data)
