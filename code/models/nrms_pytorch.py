import torch
import torch.nn as nn
import torch.optim as optim


# NRMS Model
class NRMSModel(nn.Module):
    def __init__(self, embedding_dim, history_size, npratio):
        super(NRMSModel, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        self.history_size = history_size
        self.embedding_dim = embedding_dim
        self.npratio = npratio
        self.final_fc = nn.Linear(embedding_dim, 1)  # Prediction layer

    def forward(self, his_input_title, pred_input_title):
        # Ensure inputs are float32
        his_input_title = his_input_title.float()
        pred_input_title = pred_input_title.float()

        # Attention over history articles
        attn_weights = self.attention(his_input_title)  # Shape: (batch_size, history_size, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        user_representation = torch.sum(his_input_title * attn_weights, dim=1)  # Shape: (batch_size, embedding_dim)

        # Dot product with candidate articles
        scores = torch.bmm(pred_input_title, user_representation.unsqueeze(2)).squeeze(2)  # (batch_size, npratio)
        return scores