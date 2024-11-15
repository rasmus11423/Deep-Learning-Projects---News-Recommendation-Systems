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


from transformers import AutoTokenizer

# Load a tokenizer for a specific model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Test tokenization
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)


import numpy as np
import polars as pl
from tensorflow.keras.utils import Sequence

# Mock NRMSDataLoader class
class MockNRMSDataLoader(Sequence):
    def __init__(self, num_samples, batch_size, input_shape, output_shape):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.input_shape = input_shape  # Shape for his_input_title and pred_input_title
        self.output_shape = output_shape  # Shape for labels

    def __len__(self):
        # Number of batches
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        # Generate synthetic data for inputs and labels
        batch_x_his_input = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        batch_x_pred_input = np.random.rand(self.batch_size, *self.input_shape).astype(np.float32)
        batch_y = np.random.randint(0, 2, size=(self.batch_size, *self.output_shape)).astype(np.float32)

        # Return a tuple of inputs and labels
        return (batch_x_his_input, batch_x_pred_input), batch_y

# Test the MockNRMSDataLoader
def test_mock_dataloader():
    # Parameters for testing
    num_samples = 105  # Total number of samples
    batch_size = 32  # Size of each batch
    input_shape = (10, 300)  # Example: history size 10, embedding dim 300
    output_shape = (1,)  # Binary labels for each sample

    # Create the Mock DataLoader
    dataloader = MockNRMSDataLoader(num_samples, batch_size, input_shape, output_shape)

    # Test __len__
    print(f"Number of batches: {len(dataloader)}")

    # Test __getitem__
    first_batch = dataloader[0]
    print("First batch inputs shape (his_input_title):", first_batch[0][0].shape)
    print("First batch inputs shape (pred_input_title):", first_batch[0][1].shape)
    print("First batch labels shape:", first_batch[1].shape)

    # Iterate through all batches
    for idx, batch in enumerate(dataloader):
        print(f"Batch {idx + 1}:")
        print("Inputs (his_input_title):", batch[0][0].shape)
        print("Inputs (pred_input_title):", batch[0][1].shape)
        print("Labels:", batch[1].shape)
        if idx == 2:  # Exit after 3 batches for testing
            break

# Run the test
test_mock_dataloader()
