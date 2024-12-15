import sys
from pathlib import Path

# Add the project directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from models.layers_pytorch import AttLayer2, SelfAttention


class NewsEncoder(nn.Module):
    def __init__(self, word2vec_embedding, title_size, dropout, head_num, head_dim, attention_hidden_dim):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(word2vec_embedding, dtype=torch.float32),
            freeze=False
        )
        self.embedding_dim = self.embedding.weight.shape[1]
        #print(f"Embedding size of word2vec initialized in NewsEncoder: {self.embedding_dim}")
        
        # Correct initialization
        self.dropout1 = nn.Dropout(dropout)
        # Fix in UserEncoder and NewsEncoder Initialization

        self.self_attention = SelfAttention(head_num, head_dim) # input: head_num 16, head_dim 16: output = 256
        
        self.dropout2 = nn.Dropout(dropout)

        self.att_layer = AttLayer2()  # input self

    def forward(self, sequences_input_title):
        
        sequences_input_title = sequences_input_title.long()
        embedded_sequences_title = self.embedding(sequences_input_title)

        #print(f"embedded_sequences_title in NewsEncoder: {embedded_sequences_title.shape}")

        y = self.dropout1(embedded_sequences_title)
        y = self.self_attention(y, y, y)

        y = self.dropout2(y)
        pred_title = self.att_layer(y)
        
        # Add debugging
        #print(f"Predicted Title Shape: {pred_title.shape}")

        return pred_title
    
class UserEncoder(nn.Module):
    def __init__(self, titleencoder, history_size, head_num, head_dim, attention_hidden_dim, seed=0):
        super(UserEncoder, self).__init__()
        self.history_size = history_size
        self.titleencoder = titleencoder
        self.self_attention = SelfAttention(multiheads=head_num, head_dim=head_dim)

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
        # if user_representation.size(-1) != self.embedding_dim:
        #     raise RuntimeError(f"UserEncoder Error: Expected embedding_dim={self.embedding_dim}, "
        #                     f"but got {user_representation.size(-1)}")
        return user_representation
    


class NRMSModel(nn.Module):
    def __init__(self, user_encoder, news_encoder):
        super(NRMSModel, self).__init__()
        self.user_encoder = user_encoder
        self.news_encoder = news_encoder
        self.embedding_dim = news_encoder.embedding_dim

    def forward(self, his_input_title, pred_input_title, pred_input_title_one=None):
        # Correct unpacking of shape
        batch_size, effective_batch_size, title_size = pred_input_title.size()

        # Encode user history
        user_representation = self.user_encoder(his_input_title)

        # Flatten and encode candidate articles
        pred_input_title = pred_input_title.view(-1, title_size)
        #print(f"Flattened Candidate Articles Shape: {pred_input_title.shape}")
        news_present = self.news_encoder(pred_input_title)

        # Reshape for batch processing
        news_present = news_present.view(batch_size, effective_batch_size, news_present.size(-1))
        #print(f"Reshaped News Present Shape: {news_present.shape}")

        # Compute predictions
        preds = torch.bmm(news_present, user_representation.unsqueeze(2)).squeeze(2)

        pred_one = None
        if pred_input_title_one is not None:
            pred_input_title_one = pred_input_title_one.squeeze(1)
            news_present_one = self.news_encoder(pred_input_title_one)
            pred_one = torch.bmm(news_present_one.unsqueeze(1), user_representation.unsqueeze(2)).squeeze(2)
            pred_one = torch.sigmoid(pred_one)

        return preds, pred_one
