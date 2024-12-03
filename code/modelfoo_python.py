import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.dataloader_pytorch import NRMSDataLoader
from models.layers_pytorch import SelfAttention, AttLayer2
from models.nrms_pytorch import NRMSModel, UserEncoder, NewsEncoder
import numpy as np
from pathlib import Path
import polars as pl
from utils._behaviors import create_binary_labels_column
from utils._articles import create_article_id_to_value_mapping
from transformers import AutoTokenizer, AutoModel
from utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_USER_COL
    )


def get_word2vec_embedding_from_tokenizer(tokenizer_path, model_path, embedding_dim):
    """
    Generate word2vec embeddings using a pre-trained tokenizer and model.

    Args:
        tokenizer_path (str or Path): Path to the pre-trained tokenizer.
        model_path (str or Path): Path to the pre-trained model.
        embedding_dim (int): Embedding dimension.

    Returns:
        numpy.ndarray: Word embeddings for the tokenizer vocabulary.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModel.from_pretrained(model_path)

    vocab_size = len(tokenizer)
    model_dim = model.config.hidden_size  # Get the embedding dimension from the model
    linear_projection = nn.Linear(model_dim, embedding_dim)  # Linear layer for projection
    linear_projection.eval()  # Ensure no gradients are computed for projection

    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for token_id in range(vocab_size):
        token_tensor = torch.tensor([token_id]).unsqueeze(0)  # Shape: (1, 1)
        with torch.no_grad():
            output = model.embeddings.word_embeddings(token_tensor).squeeze(0)
            projected_output = linear_projection(output)  # Project to the desired dimension
        embedding_matrix[token_id] = projected_output.numpy()

    return embedding_matrix

# Training Function
def train_model(train_dataloader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for (his_input_title, pred_input_title), labels in train_dataloader:
        his_input_title = his_input_title.to(device)
        pred_input_title = pred_input_title.to(device)
        labels = labels.argmax(dim=1).to(device)

        optimizer.zero_grad()
        preds, _ = model(his_input_title, pred_input_title)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(preds, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    return train_loss, train_acc

# Validation Function
def validate_model(val_dataloader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for (his_input_title, pred_input_title), labels in val_dataloader:
            his_input_title = his_input_title.to(device)
            pred_input_title = pred_input_title.to(device)
            labels = labels.argmax(dim=1).to(device)

            preds, _ = model(his_input_title, pred_input_title)
            loss = criterion(preds, labels)

            val_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    return val_loss, val_acc

# Main Script
if __name__ == "__main__":
    print("running with the tokenizer")

    # Paths to dataset
    BASE_PATH = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_PATH.joinpath("data")
    LOCAL_TOKENIZER_PATH = DATA_PATH.joinpath("local-tokenizer")
    LOCAL_MODEL_PATH = DATA_PATH.joinpath("local-tokenizer-model")
    # Constants
    TOKEN_COL = "tokens"
    N_SAMPLES = "n"
    BATCH_SIZE = 32
    title_size = 30  # Match the word-level token size
    embedding_dim = 128  # Match embedding dimensions with UserEncoder and NewsEncoder
    history_size = 3
    npratio = 5
    head_num = 8
    head_dim = 16
    attention_hidden_dim = 200
    dropout = 0.2  # Dropout rate

    # Prepare Articles Data
    df_articles = (
        pl.scan_parquet(DATA_PATH.joinpath("ebnerd_demo", "articles.parquet"))
        .select(["article_id", "category", "sentiment_label"])
        .with_columns(pl.Series(TOKEN_COL, np.random.rand(11777, title_size)))  # Simulated tokens
        .collect()
    )

    # Prepare User History Data
    df_history = (
        pl.scan_parquet(DATA_PATH.joinpath("ebnerd_demo", "train", "history.parquet"))
        .select(["user_id", DEFAULT_HISTORY_ARTICLE_ID_COL])
        .with_columns(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(history_size))
    )

    # Prepare Behaviors Data
    df_behaviors = (
        pl.scan_parquet(DATA_PATH.joinpath("ebnerd_demo", "train", "behaviors.parquet"))
        .select([DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL])
        .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias(N_SAMPLES))
        .join(df_history, on=DEFAULT_USER_COL, how="left")
        .collect()
        .pipe(create_binary_labels_column)
    )

    # Initialize word embeddings
    print("grabbed tokenizer no issue i guess")

    word2vec_embedding = get_word2vec_embedding_from_tokenizer(LOCAL_TOKENIZER_PATH, LOCAL_MODEL_PATH, embedding_dim)

    # Create Article Mappings
    article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=TOKEN_COL)

    # Prepare Train and Validation Sets
    df_behaviors_train = df_behaviors.filter(pl.col(N_SAMPLES) == pl.col(N_SAMPLES).min())

    # Initialize Dataloaders
    train_dataloader = NRMSDataLoader(
        behaviors=df_behaviors_train,
        article_dict=article_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        unknown_representation="zeros",
        eval_mode=False,
        batch_size=BATCH_SIZE,
    )
    val_dataloader = NRMSDataLoader(
        behaviors=df_behaviors,
        article_dict=article_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        unknown_representation="zeros",
        eval_mode=True,
        batch_size=BATCH_SIZE,
    )

    print("we now have data and tokenizer")

    # Wrap in PyTorch DataLoader
    train_loader = DataLoader(train_dataloader, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_dataloader, batch_size=None, shuffle=False)

    # Initialize User and News Encoders
    news_encoder = NewsEncoder(
        word2vec_embedding=word2vec_embedding,
        title_size=title_size,
        embedding_dim=embedding_dim,
        dropout=dropout,
        head_num=head_num,
        head_dim=head_dim,
        attention_hidden_dim=attention_hidden_dim,
    )
    user_encoder = UserEncoder(
        titleencoder=news_encoder,  # Share NewsEncoder for UserEncoder
        history_size=history_size,
        embedding_dim=embedding_dim,
        head_num=head_num,
        head_dim=head_dim,
        attention_hidden_dim=attention_hidden_dim,
    )

    # Initialize NRMS Model
    model = NRMSModel(
        user_encoder=user_encoder,
        news_encoder=news_encoder,
        embedding_dim=embedding_dim,
    )

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training and Validation Loop
    epochs = 5
    for epoch in range(epochs):
        train_loss, train_acc = train_model(train_loader, model, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        val_loss, val_acc = validate_model(val_loader, model, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
