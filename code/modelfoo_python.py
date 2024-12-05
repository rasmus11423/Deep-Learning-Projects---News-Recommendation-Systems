import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pathlib import Path
import polars as pl
from tqdm import tqdm
from models.dataloader_pytorch import NRMSDataLoader
from models.nrms_pytorch import NRMSModel, UserEncoder, NewsEncoder
from utils._behaviors import create_binary_labels_column
from utils._articles import create_article_id_to_value_mapping, convert_text2encoding_with_transformers
from utils._nlp import get_transformers_word_embeddings
from utils._polars import concat_str_columns
from utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_USER_COL,
)

def train_model(train_dataloader, model, criterion, optimizer, device):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for (his_input_title, pred_input_title), labels in train_dataloader:
        # Remove unnecessary singleton dimension
        his_input_title = his_input_title.squeeze(1)  # shape: [batch_size, history_size, title_size]
        pred_input_title = pred_input_title.squeeze(1)  # shape: [batch_size, npratio, title_size]
        labels = labels.squeeze(1)  # shape: [batch_size, npratio]

        # Move data to the target device
        his_input_title = his_input_title.to(device)
        pred_input_title = pred_input_title.to(device)
        labels = labels.argmax(dim=1).to(device)

        # Zero gradients, forward pass, compute loss, backpropagate, and update weights
        optimizer.zero_grad()
        preds, _ = model(his_input_title, pred_input_title)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        # Update metrics
        train_loss += loss.item()
        correct += (torch.max(preds, 1)[1] == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    return train_loss, train_acc

def validate_model(val_dataloader, model, loss_function, device):
    """
    Validate the model on a validation dataset.
    """
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in val_dataloader:
            # Unpack batch (modify this depending on DataLoader output)
            (his_input_title, pred_input_title), labels = batch
            # Remove unnecessary singleton dimension
            his_input_title = his_input_title.squeeze(1)  # shape: [batch_size, history_size, title_size]
            pred_input_title = pred_input_title.squeeze(1)  # shape: [batch_size, npratio, title_size]
            labels = labels.squeeze(1)  # shape: [batch_size, npratio]

            # Move data to the target device
            his_input_title = his_input_title.to(device)
            pred_input_title = pred_input_title.to(device)
            labels = labels.argmax(dim=1).to(device)

            print(f"preds: {preds}")
            print(f"labels: {labels}")

            preds, _ = model(his_input_title, pred_input_title)
            loss = criterion(preds, labels)
            val_loss += loss.item()

            # Compute accuracy: Get predicted class by taking the argmax over logits
            predicted_classes = torch.argmax(preds, dim=1)

            # Compute correct predictions
            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_dataloader)
    val_acc = correct / total

    return val_loss, val_acc



def load_data(data_path, title_size, embedding_dim, history_size, tokenizer_path, model_path):

    TEXT_COLUMNS_TO_USE = ["article_id", "category", "sentiment_label"]

    print("Loading articles and generating embeddings...")
    # Load transformer model and tokenizer
    transformer_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    transformer_model = AutoModel.from_pretrained(model_path)
    
    # Initialize word embeddings
    word2vec_embedding = get_transformers_word_embeddings(transformer_model)

    print("Tokenizer loaded sucessfully")

    df_articles = (
        pl.scan_parquet(data_path / "ebnerd_demo/articles.parquet")
        .select(["article_id", "category", "sentiment_label"])
        .collect()
    )

    # Process articles
    df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)

    df_articles, token_col_title = convert_text2encoding_with_transformers(
        df_articles, transformer_tokenizer, cat_cal, max_length=title_size
    )

    num_rows = df_articles.height  # Correctly define num_rows
    df_articles = df_articles.with_columns(
        pl.Series("tokens", np.random.rand(num_rows, title_size))
    )


    df_history = (
        pl.scan_parquet(data_path / "ebnerd_demo/train/history.parquet")
        .select(["user_id", DEFAULT_HISTORY_ARTICLE_ID_COL])
        .with_columns(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(history_size))
    )

    df_behaviors_test = (
        pl.scan_parquet(data_path / "ebnerd_demo/train/behaviors.parquet")
        .select([DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL])
        .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias("n"))
        .join(df_history, on=DEFAULT_USER_COL, how="left")
        .collect()
        .pipe(create_binary_labels_column)
    )

    df_behaviors_validation = (
        pl.scan_parquet(data_path / "ebnerd_demo/validation/behaviors.parquet")
        .select([DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL])
        .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias("n"))
        .join(df_history, on=DEFAULT_USER_COL, how="left")
        .collect()
        .pipe(create_binary_labels_column)
    )

    article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=token_col_title)

    return df_behaviors_test, df_behaviors_validation, article_mapping, word2vec_embedding

def load_test_data(data_path, title_size):
    """
    Load test data while limiting the dataset to the first 10 rows for testing.

    Args:
        data_path: Path to the data directory.
        title_size: Size of the title token embeddings.

    Returns:
        df_behaviors: The behaviors dataframe.
        article_mapping: Mapping of article IDs to token embeddings.
    """
    # Load and preprocess test articles
    df_articles = (
        pl.scan_parquet(data_path / "ebnerd_testset/articles.parquet")
        .select(["article_id", "category"])
        .collect()
        .head(10)
    )
    num_rows = df_articles.height
    df_articles = df_articles.with_columns(
        pl.Series("tokens", np.random.rand(num_rows, title_size))
    )

    # Load and preprocess test user history
    df_history = (
        pl.scan_parquet(data_path / "ebnerd_testset/test/history.parquet")
        .select([DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL])
        .collect()
        .head(10)
        .with_columns(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(3))
    )

    # Load and preprocess test behaviors
    behaviors_path = data_path / "ebnerd_testset/test/behaviors.parquet"
    behaviors_columns = pl.scan_parquet(behaviors_path).collect_schema().names()

    if DEFAULT_CLICKED_ARTICLES_COL not in behaviors_columns:
        df_behaviors = (
            pl.scan_parquet(behaviors_path)
            .select([DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL])
            .collect()
            .head(10)
            .with_columns(
                pl.Series(DEFAULT_CLICKED_ARTICLES_COL, [[] for _ in range(10)])
            )
            .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias("n"))
            .join(df_history, on=DEFAULT_USER_COL, how="left")
            .pipe(create_binary_labels_column)
        )
    else:
        df_behaviors = (
            pl.scan_parquet(behaviors_path)
            .select([DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL])
            .collect()
            .head(10)
            .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias("n"))
            .join(df_history, on=DEFAULT_USER_COL, how="left")
            .pipe(create_binary_labels_column)
        )

    # Map articles to token embeddings
    article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col="tokens")
    return df_behaviors, article_mapping

def initialize_model(word2vec_embedding, title_size, embedding_dim, history_size, head_num, head_dim, attention_hidden_dim, dropout):
    """
    Initialize the NRMS model with consistent architecture for training and evaluation.
    """
    print("Initializing NewsEncoder and UserEncoder with all submodules...")

    # Initialize NewsEncoder with self_attention and att_layer
    news_encoder = NewsEncoder(
        word2vec_embedding=word2vec_embedding,
        title_size=title_size,
        embedding_dim=embedding_dim,
        dropout=dropout,
        head_num=head_num,
        head_dim=head_dim,
        attention_hidden_dim=attention_hidden_dim,
    )

    # Initialize UserEncoder with self_attention and att_layer
    user_encoder = UserEncoder(
        titleencoder=news_encoder,
        history_size=history_size,
        embedding_dim=embedding_dim,
        head_num=head_num,
        head_dim=head_dim,
        attention_hidden_dim=attention_hidden_dim,
    )

    # Combine into NRMSModel
    model = NRMSModel(
        user_encoder=user_encoder,
        news_encoder=news_encoder,
        embedding_dim=embedding_dim,
    )

    print("Model initialized with the following architecture:")
    print(model)

    return model

def evaluate_model(test_loader, model, device):
    """
    Evaluate the model on test data.

    Args:
        test_loader: Test DataLoader.
        model: Model to evaluate.
        device: Device (CPU or GPU).

    Returns:
        accuracy (float): Test accuracy.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for (his_input_title, pred_input_title), labels in test_loader:
            # Move data to the target device
            his_input_title = his_input_title.to(device)
            pred_input_title = pred_input_title.to(device)
            labels = labels.argmax(dim=1).to(device)

            # Forward pass and update metrics
            preds, _ = model(his_input_title, pred_input_title)
            correct += (torch.max(preds, 1)[1] == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# Main Script
if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Paths and constants
    BASE_PATH = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_PATH.joinpath("data")
    LOCAL_TOKENIZER_PATH = DATA_PATH.joinpath("local-tokenizer")
    LOCAL_MODEL_PATH = DATA_PATH.joinpath("local-tokenizer-model")
    BATCH_SIZE = 16
    N_SAMPLES = "n"
    #TODO: implement padding for history so we can history size bigger than 1
    title_size, embedding_dim, history_size = 30, 128, 1
    head_num, head_dim, attention_hidden_dim, dropout = 8, 16, 200, 0.2

    # Load data
    df_behaviors_train, df_behaviors_validation, article_mapping, word2vec_embedding = load_data(
        DATA_PATH, title_size, embedding_dim, history_size, LOCAL_TOKENIZER_PATH, LOCAL_MODEL_PATH
    )

    print("we now have data and tokenizer")

    # Initialize dataloaders
    # Prepare Train and Validation Sets
    df_behaviors_train = df_behaviors_train.filter(pl.col(N_SAMPLES) == pl.col(N_SAMPLES).min())
    df_behaviors_validation = df_behaviors_validation.filter(pl.col(N_SAMPLES) == pl.col(N_SAMPLES).min())

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
        behaviors=df_behaviors_validation,
        article_dict=article_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        unknown_representation="zeros",
        eval_mode=True,
        batch_size=BATCH_SIZE,
    )

    # Wrap in PyTorch DataLoader
    train_loader = DataLoader(train_dataloader, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataloader, batch_size=BATCH_SIZE, shuffle=False)

    
    print("Dataloder for train/val successful")

    # Initialize model
    model = initialize_model(
        word2vec_embedding, title_size, embedding_dim, history_size, head_num, head_dim, attention_hidden_dim, dropout
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    model.to(device)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training and validation loop
    epochs = 3
    for epoch in range(epochs):
        # Train the model
        with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}") as pbar:
            train_loss, train_acc = train_model(pbar, model, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}")

        # Validate the model
        with tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}") as pbar:
            val_loss, val_acc = validate_model(pbar, model, criterion, device)
        print(f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

    torch.cuda.empty_cache()
    # Load and evaluate on test data
""" df_test_behaviors, test_article_mapping = load_test_data(DATA_PATH, title_size)

    test_dataloader = NRMSDataLoader(
            behaviors=df_test_behaviors,
            article_dict=test_article_mapping,
            history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            unknown_representation="zeros",
            eval_mode=True,
            batch_size=1,
        )
    test_loader = DataLoader(test_dataloader, batch_size=None, shuffle=False) """

print("Done")