import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers_pytorch import SelfAttention
from models.layers_pytorch import AttLayer2


class UserEncoder(nn.Module):
    def __init__(self, titleencoder, history_size, embedding_dim, head_num, head_dim, attention_hidden_dim, seed=0):
        super(UserEncoder, self).__init__()
        self.history_size = history_size
        self.embedding_dim = embedding_dim
        self.titleencoder = titleencoder
        self.self_attention = SelfAttention(multiheads=head_num, head_dim=head_dim, seed=seed)
        self.attention = AttLayer2(dim=attention_hidden_dim, seed=seed)

    def forward(self, his_input_title):
        """
        Forward pass for the UserEncoder.

        Args:
            his_input_title (torch.Tensor): Input tensor of shape (batch_size, history_size, title_size).

        Returns:
            torch.Tensor: User representation of shape (batch_size, embedding_dim).
        """
        batch_size, history_size, _ = his_input_title.size()

        # Convert to float32
        his_input_title = his_input_title.float()

        # Apply titleencoder in a TimeDistributed manner
        his_input_title = his_input_title.view(-1, his_input_title.size(-1))  # Flatten to (batch_size * history_size, title_size)
        click_title_presents = self.titleencoder(his_input_title)  # (batch_size * history_size, embedding_dim)
        click_title_presents = click_title_presents.view(batch_size, history_size, -1)  # Reshape back to (batch_size, history_size, embedding_dim)

        # Apply self-attention
        self_attention_out = self.self_attention(click_title_presents, click_title_presents, click_title_presents)

        # Apply soft alignment attention (AttLayer2)
        user_representation = self.attention(self_attention_out)  # (batch_size, embedding_dim)

        # Validate user_representation dimensions
        assert user_representation.size(1) == self.embedding_dim, \
            f"Expected embedding_dim={self.embedding_dim}, but got {user_representation.size(1)}"

        return user_representation

class NRMSModel(nn.Module):
    def __init__(self, titleencoder, embedding_dim, history_size, npratio, head_num, head_dim, attention_hidden_dim, seed=0):
        super(NRMSModel, self).__init__()
        self.user_encoder = UserEncoder(
            titleencoder=titleencoder,
            history_size=history_size,
            embedding_dim=embedding_dim,
            head_num=head_num,
            head_dim=head_dim,
            attention_hidden_dim=attention_hidden_dim,
            seed=seed,
        )
        self.embedding_dim = embedding_dim

    def forward(self, his_input_title, pred_input_title):
        """
        Forward pass for the NRMS model.

        Args:
            his_input_title (torch.Tensor): User history input (batch_size, history_size, title_size).
            pred_input_title (torch.Tensor): Candidate articles (batch_size, npratio, embedding_dim).

        Returns:
            torch.Tensor: Prediction scores (batch_size, npratio).
        """
        # Get user representation
        user_representation = self.user_encoder(his_input_title)  # (batch_size, embedding_dim)

        # Validate pred_input_title dimensions
        assert pred_input_title.size(2) == self.embedding_dim, \
            f"Expected pred_input_title last dim={self.embedding_dim}, but got {pred_input_title.size(2)}"

        # Dot product with candidate articles
        scores = torch.bmm(pred_input_title, user_representation.unsqueeze(2)).squeeze(2)  # (batch_size, npratio)
        return scores
