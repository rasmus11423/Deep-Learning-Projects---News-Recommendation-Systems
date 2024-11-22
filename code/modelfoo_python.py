import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.dataloader_pytorch import NRMSDataLoader
import numpy as np
from pathlib import Path
import polars as pl
from utils._behaviors import create_binary_labels_column
from utils._articles import create_article_id_to_value_mapping
from utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_USER_COL,
)
from models.nrms_pytorch import NRMSModel

# Training and Validation Loops
def train_and_validate(train_dataloader, val_dataloader, model, epochs=5, lr=0.001, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Optimizer

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for (his_input_title, pred_input_title), labels in train_dataloader:
            his_input_title = his_input_title.to(device).float()  # Ensure float32
            pred_input_title = pred_input_title.to(device).float()  # Ensure float32
            labels = labels.argmax(dim=1).to(device)  # Convert one-hot to class indices

            optimizer.zero_grad()
            outputs = model(his_input_title, pred_input_title)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for (his_input_title, pred_input_title), labels in val_dataloader:
                his_input_title = his_input_title.to(device).float()  # Ensure float32
                pred_input_title = pred_input_title.to(device).float()  # Ensure float32
                labels = labels.argmax(dim=1).to(device)

                outputs = model(his_input_title, pred_input_title)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
print("I am here")
# Main Script
if __name__ == "__main__":
    # Paths to dataset
    BASE_PATH = Path(__file__).resolve().parent
    DATA_PATH = BASE_PATH.joinpath("data")  # Adjust as per data structure

    # Constants
    TOKEN_COL = "tokens"
    N_SAMPLES = "n"
    BATCH_SIZE = 32

    # Prepare Articles Data
    df_articles = (
        pl.scan_parquet(DATA_PATH.joinpath("ebnerd_demo", "articles.parquet"))
        .select(["article_id", "category", "sentiment_label"])
        .with_columns(pl.Series(TOKEN_COL, np.random.rand(11777, 50)))  # Random embeddings
        .collect()
    )

    # Prepare User History Data
    df_history = (
        pl.scan_parquet(DATA_PATH.joinpath("ebnerd_demo", "train", "history.parquet"))
        .select(["user_id", DEFAULT_HISTORY_ARTICLE_ID_COL])
        .with_columns(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(3))  # Limit to last 3 articles
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

    # Wrap in PyTorch DataLoader
    train_loader = DataLoader(train_dataloader, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_dataloader, batch_size=None, shuffle=False)

    # Initialize the model
    embedding_dim = 50
    history_size = 3
    npratio = 5
    model = NRMSModel(embedding_dim, history_size, npratio)

    # Train and validate
    train_and_validate(train_loader, val_loader, model, epochs=5, lr=0.001, device="cuda")
