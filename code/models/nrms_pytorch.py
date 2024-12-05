import sys
from pathlib import Path

# Add the project directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.layers_pytorch import SelfAttention
from models.layers_pytorch import AttLayer2,SelfAttention


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
        # Move input tensor to the correct device
        his_input_title = his_input_title.to(next(self.parameters()).device)

        # Convert input to LongTensor for embedding lookup (if needed)
        his_input_title = his_input_title.long()

        batch_size, history_size, _ = his_input_title.size()

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



class NewsEncoder(nn.Module):
    def __init__(self, word2vec_embedding, title_size, embedding_dim, dropout, head_num, head_dim, attention_hidden_dim):
        """
        Initializes the News Encoder for NRMS.

        Args:
            word2vec_embedding (np.ndarray): Pre-trained word embeddings.
            title_size (int): Size of the title sequence.
            embedding_dim (int): Dimension of embeddings.
            dropout (float): Dropout rate.
            head_num (int): Number of attention heads.
            head_dim (int): Dimension of each attention head.
            attention_hidden_dim (int): Hidden dimension for the attention layer.
        """
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(word2vec_embedding, dtype=torch.float32),
            freeze=False  # Trainable embeddings
        )
        self.dropout1 = nn.Dropout(dropout)
        self.self_attention = SelfAttention(multiheads=head_num, head_dim=head_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.att_layer = AttLayer2(dim=attention_hidden_dim)

    def forward(self, sequences_input_title ):
        """
        Forward pass of the News Encoder.

        Args:
            sequences_input_title (torch.Tensor): Input tensor of shape (batch_size, title_size).

        Returns:
            torch.Tensor: Encoded news representation of shape (batch_size, embedding_dim).
        """

        # Ensure input is of type LongTensor for embedding lookup
        sequences_input_title = sequences_input_title.long()  # Convert to LongTensor

        # Embedding lookup
        embedded_sequences_title = self.embedding(sequences_input_title)  # Shape: (batch_size, title_size, embedding_dim)

        # Apply first dropout
        y = self.dropout1(embedded_sequences_title)

        # Apply self-attention
        y = self.self_attention(y, y, y)  # Shape: Q, K, V

        # Apply second dropout
        y = self.dropout2(y)

        # Apply attention layer to get the final news representation
        pred_title = self.att_layer(y)  # Shape: (batch_size, embedding_dim)

        return pred_title
    

class NRMSModel(nn.Module):
    def __init__(self, user_encoder, news_encoder, embedding_dim):
        """
        NRMS Model initialization.

        Args:
            user_encoder (nn.Module): Instance of the UserEncoder.
            news_encoder (nn.Module): Instance of the NewsEncoder.
            embedding_dim (int): Dimension of the embeddings.
        """
        super(NRMSModel, self).__init__()
        self.user_encoder = user_encoder
        self.news_encoder = news_encoder
        self.embedding_dim = embedding_dim

    def forward(self, his_input_title, pred_input_title, pred_input_title_one=None):
        """
        Forward pass for the NRMS model.

        Args:
            his_input_title (torch.Tensor): User history input of shape (batch_size, history_size, title_size).
            pred_input_title (torch.Tensor): Candidate articles of shape (batch_size, npratio, title_size).
            pred_input_title_one (torch.Tensor, optional): A single candidate article of shape (batch_size, 1, title_size).

        Returns:
            tuple: Predictions for training and single candidate article.
        """
        batch_size, npratio, title_size = pred_input_title.size()

        # Encode user history into user representation
        user_representation = self.user_encoder(his_input_title)  # (batch_size, embedding_dim)

        # Encode candidate articles
        pred_input_title = pred_input_title.view(-1, title_size)  # Flatten to (batch_size * npratio, title_size)
        news_present = self.news_encoder(pred_input_title)  # (batch_size * npratio, embedding_dim)
        news_present = news_present.view(batch_size, npratio, self.embedding_dim)  # Reshape back

        # Compute predictions for training
        preds = torch.bmm(news_present, user_representation.unsqueeze(2)).squeeze(2)  # (batch_size, npratio)
        #preds = F.softmax(preds, dim=-1)  # Apply softmax activation

        # Compute predictions for a single candidate article
        pred_one = None
        if pred_input_title_one is not None:
            pred_input_title_one = pred_input_title_one.squeeze(1)  # (batch_size, title_size)
            news_present_one = self.news_encoder(pred_input_title_one)  # (batch_size, embedding_dim)
            pred_one = torch.bmm(news_present_one.unsqueeze(1), user_representation.unsqueeze(2)).squeeze(2)  # (batch_size, 1)
            pred_one = torch.sigmoid(pred_one)  # Apply sigmoid activation

        return preds, pred_one