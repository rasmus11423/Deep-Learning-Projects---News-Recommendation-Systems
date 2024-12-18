import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pathlib import Path
import polars as pl
import argparse
import neptune
import os
from tqdm import tqdm
from models.evaluation import auc_score_custom
from models.dataloader_pytorch import NRMSDataLoader
from models.nrms_pytorch import NRMSModel, UserEncoder, NewsEncoder
from utils._behaviors import create_binary_labels_column,truncate_history,sampling_strategy_wu2019
from utils._articles import create_article_id_to_value_mapping, convert_text2encoding_with_transformers
from utils._nlp import get_transformers_word_embeddings
from utils._polars import concat_str_columns, slice_join_dataframes
from utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_USER_COL,
    DEFAULT_IMPRESSION_ID_COL
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model with optional Neptune logging.")
parser.add_argument("--debug", action="store_true", help="Run the script in debug mode (no Neptune logging).")
parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use for training.")
args = parser.parse_args()

# Access the dataset name
dataset_name = args.dataset
print(f"Using dataset: {dataset_name}")

# Initialize Neptune run unless in debug mode
if not args.debug:
    # Get API token from environment variable
    api_token = os.getenv("NEPTUNE_API_TOKEN")
    if api_token is None:
        raise ValueError("Neptune API token not found. Please set the 'NEPTUNE_API_TOKEN' environment variable.")


    run = neptune.init_run(
        project="Deep-learning-project/model-training",
        api_token=api_token,
        )

    print("neptune intialized")
else:
    print("Debug mode: Neptune logging is disabled.")

def train_model(train_dataloader, model, criterion, optimizer, device):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []

    for (his_input_title, pred_input_title), labels in train_dataloader:
        # Remove unnecessary singleton dimension
        his_input_title = his_input_title.squeeze(1)  # shape: [batch_size, history_size, title_size]
        pred_input_title = pred_input_title.squeeze(1)  # shape: [batch_size, npratio, title_size]
        labels = labels.squeeze(1)  # shape: [batch_size, npratio]
        if his_input_title.ndim == 4:
            his_input_title = his_input_title.squeeze(1)
        if pred_input_title.ndim == 4:
            pred_input_title = pred_input_title.squeeze(1)
        if labels.ndim == 3:
            labels = labels.squeeze(1)


        # Move data to the target device
        his_input_title = his_input_title.to(device)
        pred_input_title = pred_input_title.to(device)
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)
        labels = labels.to(device)


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

        # Store predictions and labels for AUC calculation
        all_preds.extend(preds.softmax(dim=1)[:, 1].detach().cpu().numpy())  # Assuming positive class is at index 1
        all_labels.extend(labels.cpu().numpy())

    train_acc = correct / total

    # Convert to NumPy arrays for the custom AUC function
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate AUC score using the custom implementation
    try:
        train_auc = auc_score_custom(all_labels, all_preds)
    except Exception as e:
        train_auc = float('nan')  # Handle potential errors gracefully
        print(f"AUC calculation failed: {e}")

    if not args.debug:
        run["train/loss"].log(train_loss / len(train_dataloader))
        run["train/accuracy"].log(train_acc)
        run["train/auc"].log(train_auc)
    
    print(f"History size in dataloader: {his_input_title.shape}, NPRatio: {pred_input_title.shape}")
    
    return train_loss, train_acc, train_auc


def validate_model(val_dataloader, model, criterion, device):
    """
    Validate the model on a validation dataset.
    """
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Unpack batch (modify this depending on DataLoader output)
            (his_input_title, pred_input_title), labels = batch

            if his_input_title.ndim > 3:
                his_input_title = his_input_title.squeeze(2)
            if pred_input_title.ndim > 3:
                pred_input_title = pred_input_title.squeeze(2)
            if labels.ndim > 2:
                labels = labels.squeeze(-1)
            
            # Reduce labels to class indices (if one-hot encoded)
            labels = labels.argmax(dim=1).to(device)

            # Move data to the target device
            his_input_title = his_input_title.to(device)
            pred_input_title = pred_input_title.to(device)

            # Forward pass
            preds, _ = model(his_input_title, pred_input_title)

            # Apply softmax if outputs are logits
            preds = torch.softmax(preds, dim=1)

            loss = criterion(preds, labels)

            # Update loss and accuracy
            val_loss += loss.item()
            correct += (torch.max(preds, 1)[1] == labels).sum().item()
            total += labels.size(0)

            # Store predictions and labels for AUC calculation
            all_preds.extend(preds[:, 1].detach().cpu().numpy())  # Assuming positive class is at index 1
            all_labels.extend(labels.cpu().numpy())

    # Compute validation metrics
    val_acc = correct / total if total > 0 else 0
    val_loss /= len(val_dataloader)

    # Convert to NumPy arrays for the custom AUC function
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate AUC score using the custom implementation
    try:
        val_auc = auc_score_custom(all_labels, all_preds)
    except Exception as e:
        val_auc = float('nan')  # Handle potential errors gracefully
        print(f"AUC calculation failed: {e}")

    if not args.debug:
        run["validation/loss"].log(val_loss)
        run["validation/accuracy"].log(val_acc)
        run["validation/auc"].log(val_auc)

    return val_loss, val_acc, val_auc

def load_data(data_path, title_size, history_size, tokenizer_path, model_path):

    def ebnerd_from_path(path: str, history_size: int = 30) -> pl.LazyFrame:
        """
        Load ebnerd - function
        """

        set_name = path+"/history.parquet"
        history_path = data_path /set_name

        df_history = (
            pl.scan_parquet(history_path)
            .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
            .pipe(
                truncate_history,
                column=DEFAULT_HISTORY_ARTICLE_ID_COL,
                history_size=history_size,
                padding_value=0,
                enable_warning=False,
            )
        ).lazy()

        set_name = path+"/behaviors.parquet"
        bahoir_path = data_path /set_name

        df_behaviors = (
            pl.scan_parquet(bahoir_path)
            .collect()
            .pipe(
                slice_join_dataframes,
                df2=df_history.collect(),
                on=DEFAULT_USER_COL,
                how="left",
            )
        )
        return df_behaviors

    def load_history(path: Path, history_size: int = 30) -> pl.LazyFrame:
        """
        Load ebnerd - function
        """
        df_history = (
            pl.scan_parquet(data_path / "train/history.parquet")
            .select(["user_id", DEFAULT_HISTORY_ARTICLE_ID_COL])
            .with_columns(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(history_size))
            #.select(["user_id", DEFAULT_HISTORY_ARTICLE_ID_COL])
            # .collect()
            .pipe(
                truncate_history,
                column=DEFAULT_HISTORY_ARTICLE_ID_COL,
                history_size=history_size,
                padding_value=0,
                enable_warning=False,
            )
        )

        # df_behaviors = (
        #     pl.scan_parquet(path.joinpath("behaviors.parquet"))
        #     .pipe(
        #         slice_join_dataframes,
        #         df2=df_history.collect(),
        #         on=DEFAULT_USER_COL,
        #         how="left",
        #     )
        # )
        return df_history

    TEXT_COLUMNS_TO_USE = ["article_id", "category", "sentiment_label"]

    print("Loading articles and generating embeddings...")
    # Load transformer model and tokenizer
    transformer_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    transformer_model = AutoModel.from_pretrained(model_path)
    
    # Initialize word embeddings

    word2vec_embedding = get_transformers_word_embeddings(transformer_model)
    


    print("Tokenizer loaded sucessfully")

    df_articles = (
        pl.scan_parquet(data_path /"articles.parquet")
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


    # df_history = (
    #     load_history( data_path /"train/behaviors.parquet" , history_size=history_size)
    #     #pl.scan_parquet(data_path / "train/history.parquet")
    #     #.select(["user_id", DEFAULT_HISTORY_ARTICLE_ID_COL])
    #     #.with_columns(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(history_size))
    # )
    

    # #print(f"Before join: df_history shape: {df_history.shape}")
    # df_behaviors_test = (
    #     #ebnerd_from_path( data_path /"train/behaviors.parquet" , history_size=history_size)
    #     pl.scan_parquet(data_path /"train/behaviors.parquet")
    #     .select([DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL])
    #     .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias("n"))
    #     .join(df_history, on=DEFAULT_USER_COL, how="left")
    #     .collect()
    #     .pipe(create_binary_labels_column)
    # )
    # #print(f"After join: df_behaviors_train shape: {df_behaviors_test.shape}")


    # df_behaviors_validation = (
    #     #ebnerd_from_path( data_path /"validation/behaviors.parquet" , history_size=history_size)
    #     pl.scan_parquet(data_path / "validation/behaviors.parquet")
    #     .select([DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL])
    #     .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias("n"))
    #     .join(df_history, on=DEFAULT_USER_COL, how="left")
    #     .collect()
    #     .pipe(create_binary_labels_column)
    # )

    COLUMNS = [
        DEFAULT_USER_COL,
        DEFAULT_HISTORY_ARTICLE_ID_COL,
        DEFAULT_INVIEW_ARTICLES_COL,
        DEFAULT_CLICKED_ARTICLES_COL,
        DEFAULT_IMPRESSION_ID_COL]
    FRACTION = 0.01
    # Load data
    df_train = (
        ebnerd_from_path("train", history_size=history_size)
        .select(COLUMNS)
        .pipe(
            sampling_strategy_wu2019,
            npratio=4,
            shuffle=True,
            with_replacement=True,
            seed=123,
        )
        .pipe(create_binary_labels_column)
        .sample(fraction=FRACTION)
    )

    df_validation = (
        ebnerd_from_path("validation", history_size=history_size)
        .select(COLUMNS)
        .pipe(create_binary_labels_column)
        .sample(fraction=FRACTION)
    )

    article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=token_col_title)

    return df_train, df_validation, article_mapping, word2vec_embedding

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
        pl.scan_parquet(data_path.parent() / "ebnerd_testset/articles.parquet")
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
        pl.scan_parquet(data_path.parent() / "ebnerd_testset/test/history.parquet")
        .select([DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL])
        .collect()
        .head(10)
        .with_columns(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(3))
    )

    # Load and preprocess test behaviors
    behaviors_path = data_path.parent() / "ebnerd_testset/test/behaviors.parquet"
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

def initialize_model(word2vec_embedding, title_size, history_size, head_num, head_dim, attention_hidden_dim, dropout):
    """
    Initialize the NRMS model with correct embedding dimension and consistent architecture.
    """
    print("Initializing NewsEncoder and UserEncoder with all submodules...")

    # Correctly initialize NewsEncoder
    news_encoder = NewsEncoder(
        word2vec_embedding=word2vec_embedding,
        title_size=title_size,
        dropout=dropout,
        head_num=head_num,
        head_dim=head_dim,
        attention_hidden_dim=attention_hidden_dim,
    )

    # Correctly initialize UserEncoder
    user_encoder = UserEncoder(
        titleencoder=news_encoder,
        history_size=history_size,
        head_num=head_num,
        head_dim=head_dim,
        attention_hidden_dim=attention_hidden_dim,
    )

    # Combine into NRMSModel
    model = NRMSModel(
        user_encoder=user_encoder,
        news_encoder=news_encoder,
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

def custom_collate_fn_train(batch, history_len):
    padded_batch = []
    #print(batch)

    for (history, prediction), labels in batch:
        #print(f"Original history shape: {history.shape}")
        history_padded = torch.zeros((1, history_len, history.size(-1)))
        history_padded[0, :min(history.size(1), history_len)] = history[0, :history_len]
        #print(f"Padded history shape: {history_padded.shape}")
        padded_batch.append(((history_padded, prediction), labels))

    return default_collate(padded_batch)

# def custom_collate_fn_validation(batch, history_len):
#     padded_batch = []

#     for (history, prediction), labels in batch:
#         print(f"Original history shape (Validation): {history.shape}")

#         # Ensure correct dimensions for padding
#         batch_size, current_len, feature_size = history.shape
#         history_padded = torch.zeros((batch_size, history_len, feature_size))

#         # Correctly copy data into the padded history
#         max_len = min(current_len, history_len)
#         history_padded[:, :max_len, :] = history[:, :max_len, :]

#         print(f"Padded history shape (Validation): {history_padded.shape}")

#         padded_batch.append(((history_padded, prediction), labels))

#     return default_collate(padded_batch)




# Main Script
if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Paths and constants
    BASE_PATH = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_PATH.joinpath("data").joinpath(dataset_name)
    LOCAL_TOKENIZER_PATH = BASE_PATH.joinpath("data").joinpath("local-tokenizer")
    LOCAL_MODEL_PATH = BASE_PATH.joinpath("data").joinpath("local-tokenizer-model")
    BATCH_SIZE = 32
    N_SAMPLES = "n" # 1 postive 4 negative, so batchsize adabs to change 16 * npratio = 16 * 5 = 80
    
    #TODO: implement padding for history so we can history size bigger than 1
    title_size, history_size = 30, 30
    head_num, head_dim, attention_hidden_dim, dropout = 20, 20, 200, 0.2

    # Load data
    df_behaviors_train, df_behaviors_validation, article_mapping, word2vec_embedding = load_data(
        DATA_PATH, title_size, history_size, LOCAL_TOKENIZER_PATH, LOCAL_MODEL_PATH
    )

    print("we now have data and tokenizer")

    # Initialize dataloaders
    # Prepare Train and Validation Sets
    df_behaviors_train = df_behaviors_train.head(4*BATCH_SIZE)
    df_behaviors_validation = df_behaviors_validation.head(2*BATCH_SIZE)

    # Initialize Dataloaders
    train_dataloader = NRMSDataLoader(
        behaviors=df_behaviors_train,
        article_dict=article_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        unknown_representation="zeros",
        eval_mode=False,
        batch_size=BATCH_SIZE,  # Ensure this matches 16
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
    train_loader = DataLoader(
        train_dataloader, batch_size=BATCH_SIZE, shuffle=True
    )

    print(f"Expected Train Loader Batch Size: {BATCH_SIZE}")
    #print(f"Actual Train Loader Batch Size: {len(next(iter(train_loader))[0][0])}")
    val_loader = DataLoader(val_dataloader, batch_size=BATCH_SIZE, shuffle=False)

    # val_loader = DataLoader(val_dataloader, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: custom_collate_fn_validation(x, history_size)
    # )
    print("Dataloder for train/val successful")


    model = initialize_model(
        word2vec_embedding, title_size, history_size, head_num, head_dim, attention_hidden_dim, dropout
    )
    print(f"Loaded word2vec embedding shape: {word2vec_embedding.shape}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    model.to(device)

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5) # with added weight decay
    criterion = nn.CrossEntropyLoss()

    if not args.debug:
        params = {
            "optimizer": "Adam",
            "learning_rate":0.01,
            "dataset": dataset_name,
            "batchsize": BATCH_SIZE
            }
        run["parameters"] = params

    # Training and validation loop
    epochs = 10
    for epoch in range(epochs):
        # Train the model
        with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}") as pbar:
            train_loss, train_acc, train_auc = train_model(pbar, model, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Train Auc = {train_auc:.4f}")

        # Validate the model
        with tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}") as pbar:
            val_loss, val_acc, val_auc = validate_model(pbar, model, criterion, device)
        print(f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}, Val Auc = {val_auc:.4f}")



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

if not args.debug:
    run.stop()

print("Done")
