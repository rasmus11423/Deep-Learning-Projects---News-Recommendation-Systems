import sys
from pathlib import Path

# Add the project directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from models.layers_pytorch import AttLayer2, SelfAttention


class UserEncoder(nn.Module):
    def __init__(self, titleencoder, history_size, embedding_dim, head_num, head_dim, attention_hidden_dim, seed=0):
        super(UserEncoder, self).__init__()
        self.history_size = history_size
        self.embedding_dim = embedding_dim
        self.titleencoder = titleencoder
        self.self_attention = SelfAttention(multiheads=head_num, head_dim=embedding_dim // head_num)

        self.attention = AttLayer2(dim=attention_hidden_dim, seed=seed)

    def forward(self, his_input_title):
        his_input_title = his_input_title.to(next(self.parameters()).device).long()
        batch_size, history_size, title_size = his_input_title.size()

        # Encode user history
        his_input_title = his_input_title.reshape(-1, title_size)
        click_title_presents = self.titleencoder(his_input_title)
        click_title_presents = click_title_presents.reshape(batch_size, history_size, -1)

        #print(f"Click Title Presentations Shape: {click_title_presents.shape}")

        # Apply self-attention
        self_attention_out = self.self_attention(click_title_presents, click_title_presents, click_title_presents)
        #print(f"Self-Attention Output Shape (UserEncoder): {self_attention_out.shape}")

        # Apply soft alignment attention
        user_representation = self.attention(self_attention_out)
        #print(f"User Representation Shape: {user_representation.shape}")

        # Ensure correct output dimension
        if user_representation.size(-1) != self.embedding_dim:
            raise RuntimeError(f"UserEncoder Error: Expected embedding_dim={self.embedding_dim}, "
                            f"but got {user_representation.size(-1)}")
        return user_representation



class NewsEncoder(nn.Module):
    def __init__(self, word2vec_embedding, title_size, embedding_dim, dropout, head_num, head_dim, attention_hidden_dim):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(word2vec_embedding, dtype=torch.float32),
            freeze=False
        )

        # Correct initialization
        self.dropout1 = nn.Dropout(dropout)
        # Fix in UserEncoder and NewsEncoder Initialization
        self.self_attention = SelfAttention(multiheads=6, head_dim=128)  # 6 * 128 = 768
        self.dropout2 = nn.Dropout(dropout)
        self.att_layer = AttLayer2(dim=embedding_dim)  # Fix output dim

    def forward(self, sequences_input_title):
        sequences_input_title = sequences_input_title.long()
        embedded_sequences_title = self.embedding(sequences_input_title)

        y = self.dropout1(embedded_sequences_title)
        y = self.self_attention(y, y, y)
        y = self.dropout2(y)
        pred_title = self.att_layer(y)

        # Add debugging
        # print(f"Embedded Sequences Shape: {embedded_sequences_title.shape}")
        # print(f"Self-Attention Output Shape: {y.shape}")
        # print(f"Predicted Title Shape: {pred_title.shape}")

        return pred_title


class NRMSModel(nn.Module):
    def __init__(self, user_encoder, news_encoder, embedding_dim):
        super(NRMSModel, self).__init__()
        self.user_encoder = user_encoder
        self.news_encoder = news_encoder
        self.embedding_dim = embedding_dim
        self.layer_norm_pred = nn.LayerNorm(embedding_dim)
        #print(f"Model initialized with embedding_dim={self.embedding_dim}")

    def forward(self, his_input_title, pred_input_title, pred_input_title_one=None):
        #print(f"NRMS Model Input Shapes: his_input_title={his_input_title.shape}, pred_input_title={pred_input_title.shape}")

        batch_size, npratio, title_size = pred_input_title.size()

        # Encode user history
        user_representation = self.user_encoder(his_input_title)
        #print(f"User Representation Shape: {user_representation.shape}")

        # Encode candidate articles
        pred_input_title = pred_input_title.reshape(-1, title_size)
        #print(f"Flattened Candidate Articles Shape: {pred_input_title.shape}")
        news_present = self.news_encoder(pred_input_title)

        expected_size = batch_size * npratio * self.embedding_dim
        if news_present.numel() != expected_size:
            raise RuntimeError(
                f"Shape mismatch: expected {expected_size}, but got {news_present.numel()}. "
                f"Actual shape: {news_present.shape}"
            )

        news_present = news_present.reshape(batch_size, npratio, self.embedding_dim)
        #print(f"Reshaped News Present Shape: {news_present.shape}")

        user_representation = self.layer_norm_pred(user_representation)
        #print(f"Normalized User Representation Shape: {user_representation.shape}")

        # Compute predictions
        preds = torch.bmm(news_present, user_representation.unsqueeze(2)).squeeze(2)
        #print(f"Predictions Shape: {preds.shape}")

        pred_one = None
        if pred_input_title_one is not None:
            pred_input_title_one = pred_input_title_one.squeeze(1)
            news_present_one = self.news_encoder(pred_input_title_one)
            #print(f"Single News Present Shape: {news_present_one.shape}")
            pred_one = torch.bmm(news_present_one.unsqueeze(1), user_representation.unsqueeze(2)).squeeze(2)
            pred_one = torch.sigmoid(pred_one)
            #print(f"Single Prediction Shape: {pred_one.shape}")

        return preds, pred_one
