import numpy as np
import torch
from models.layers_pytorch import SelfAttention, AttLayer2
from models.nrms_pytorch import NewsEncoder, UserEncoder, NRMSModel as PyTorchNRMSModel

# Constants for test
batch_size = 4  # Test larger batch sizes
history_size = 3
title_size = 5
embedding_dim = 8
npratio = 4
head_num = 2
head_dim = 4
attention_hidden_dim = 10

# Edge case synthetic data
synthetic_his_input = np.zeros((batch_size, history_size, title_size))  # All zeros
synthetic_pred_input = np.ones((batch_size, npratio, title_size))  # All ones
max_value_input = np.full((batch_size, history_size, title_size), 9)  # Max vocab index

# Synthetic embeddings (shared across cases)
word2vec_embedding = np.random.rand(10, embedding_dim).astype(np.float32)

# Initialize PyTorch model
news_encoder = NewsEncoder(
    word2vec_embedding=word2vec_embedding,
    title_size=title_size,
    embedding_dim=embedding_dim,
    dropout=0.2,
    head_num=head_num,
    head_dim=head_dim,
    attention_hidden_dim=attention_hidden_dim,
)

user_encoder = UserEncoder(
    titleencoder=news_encoder,
    history_size=history_size,
    embedding_dim=embedding_dim,
    head_num=head_num,
    head_dim=head_dim,
    attention_hidden_dim=attention_hidden_dim,
)

pytorch_model = PyTorchNRMSModel(
    user_encoder=user_encoder,
    news_encoder=news_encoder,
    embedding_dim=embedding_dim,
)

# PyTorch Edge Case Test
for input_data, desc in zip(
    [synthetic_his_input, synthetic_pred_input, max_value_input],
    ["All zeros", "All ones", "Max values"]
):
    print(f"\nTesting PyTorch model with {desc} input...")
    torch_his_input = torch.tensor(input_data, dtype=torch.long)
    torch_pred_input = torch.tensor(synthetic_pred_input, dtype=torch.long)
    pytorch_preds, _ = pytorch_model(torch_his_input, torch_pred_input)
    print(f"PyTorch Predictions:\n{pytorch_preds.detach().numpy()}")

# Gradient Test for PyTorch
print("\nTesting PyTorch model gradients...")
labels = torch.tensor([0] * batch_size, dtype=torch.long)  # Synthetic labels
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)

# Forward pass
torch_his_input = torch.tensor(synthetic_his_input, dtype=torch.long)
torch_pred_input = torch.tensor(synthetic_pred_input, dtype=torch.long)
pytorch_preds, _ = pytorch_model(torch_his_input, torch_pred_input)

# Compute loss and backpropagate
loss = criterion(pytorch_preds, labels)
print(f"Loss: {loss.item()}")
loss.backward()  # Backpropagation
optimizer.step()  # Perform optimizer step
print("Gradients successfully computed and optimizer step performed.")


import numpy as np
import tensorflow as tf
from types import SimpleNamespace
from models.nrms import NRMSModel  # TensorFlow NRMS Model

# Constants for test
batch_size = 4  # Test larger batch sizes
history_size = 3
title_size = 5
embedding_dim = 8
npratio = 4
head_num = 2
head_dim = 4
attention_hidden_dim = 10

# Edge case synthetic data
synthetic_his_input = np.zeros((batch_size, history_size, title_size))  # All zeros
synthetic_pred_input = np.ones((batch_size, npratio, title_size))  # All ones
max_value_input = np.full((batch_size, history_size, title_size), 9)  # Max vocab index

# Synthetic embeddings
word2vec_embedding = np.random.rand(10, embedding_dim).astype(np.float32)

# Convert hparams dictionary to object
hparams = SimpleNamespace(**{
    "history_size": history_size,
    "title_size": title_size,
    "embedding_dim": embedding_dim,
    "dropout": 0.2,
    "head_num": head_num,
    "head_dim": head_dim,
    "attention_hidden_dim": attention_hidden_dim,
    "loss": "cross_entropy_loss",
    "optimizer": "adam",
    "learning_rate": 0.001,
})

# Initialize TensorFlow model
tf_model = NRMSModel(hparams=hparams, word2vec_embedding=word2vec_embedding)

# TensorFlow Edge Case Test
for input_data, desc in zip(
    [synthetic_his_input, synthetic_pred_input, max_value_input],
    ["All zeros", "All ones", "Max values"]
):
    print(f"\nTesting TensorFlow model with {desc} input...")
    if desc == "All ones" or desc == "Max values":
        # Use correct shapes for his_input and pred_input
        tf_his_input = tf.convert_to_tensor(synthetic_his_input, dtype=tf.int32)
        tf_pred_input = tf.convert_to_tensor(input_data, dtype=tf.int32)
    else:
        tf_his_input = tf.convert_to_tensor(input_data, dtype=tf.int32)
        tf_pred_input = tf.convert_to_tensor(synthetic_pred_input, dtype=tf.int32)
    tf_preds = tf_model.model([tf_his_input, tf_pred_input])
    print(f"TensorFlow Predictions:\n{tf_preds.numpy()}")

# Gradient Test for TensorFlow
print("\nTesting TensorFlow model gradients...")
with tf.GradientTape() as tape:
    tf_his_input = tf.convert_to_tensor(synthetic_his_input, dtype=tf.int32)
    tf_pred_input = tf.convert_to_tensor(synthetic_pred_input, dtype=tf.int32)
    tf_preds = tf_model.model([tf_his_input, tf_pred_input])
    labels = tf.one_hot([0] * batch_size, npratio)  # Synthetic one-hot labels
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, tf_preds))
    print(f"Loss: {loss.numpy()}")

# Compute gradients and perform an optimization step
trainable_vars = tf_model.model.trainable_variables
gradients = tape.gradient(loss, trainable_vars)
optimizer = tf.keras.optimizers.Adam(learning_rate=hparams.learning_rate)
optimizer.apply_gradients(zip(gradients, trainable_vars))
print("Gradients successfully computed and optimizer step performed.")
