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
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            try:
                # Extract batch
                (his_input_title, pred_input_title), labels = batch

                if his_input_title.size(0) != 32:
                    print(f"[WARNING] Skipping Batch {batch_idx}: Batch size not 32")
                    continue

                # Send to device
                his_input_title = his_input_title.to(device)
                pred_input_title = pred_input_title.to(device)
                labels = labels.to(device)

                # Forward pass
                preds, _ = model(his_input_title, pred_input_title)
                loss = criterion(preds, labels.float().view(-1, 1))

                # Record results
                val_loss += loss.item()
                pred_labels = torch.round(torch.sigmoid(preds))
                val_correct += (pred_labels == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

            except Exception as e:
                print(f"[ERROR] Skipping Batch {batch_idx}: {e}")
                continue

    val_acc = correct / total if total > 0 else 0
    val_loss /= len(val_dataloader) if len(val_dataloader) > 0 else float('nan')

    try:
        val_auc = auc_score_custom(np.array(all_labels), np.array(all_preds))
    except Exception as e:
        val_auc = float('nan')
        print(f"[ERROR] AUC Calculation Failed: {e}")

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

def load_test_data(data_path, title_size, history_size, tokenizer_path, model_path):
    """
    Load and process test data for the NRMS model.
    """
    TEXT_COLUMNS_TO_USE = ["article_id", "category", "sentiment_label"]

    # Load tokenizer and model
    print("Loading tokenizer and transformer model for test data...")
    transformer_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    transformer_model = AutoModel.from_pretrained(model_path)

    # Extract word embeddings
    word2vec_embedding = get_transformers_word_embeddings(transformer_model)
    print("Test Tokenizer and Model loaded successfully.")

    # Load and process articles
    articles_path = data_path / "ebnerd_testset/articles.parquet"
    df_articles = (
        pl.read_parquet(articles_path)
        .select(TEXT_COLUMNS_TO_USE)
        .head(10)
    )

    # Combine text columns and tokenize articles
    df_articles, combined_col = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
    df_articles, token_col_title = convert_text2encoding_with_transformers(
        df_articles, transformer_tokenizer, combined_col, max_length=title_size
    )

    # Ensure `article_id_fixed` exists
    df_articles = df_articles.with_columns(
        pl.col("article_id").alias("article_id_fixed"),
        pl.Series("tokens", np.random.rand(df_articles.height, title_size))
    )
    print("Test Articles Columns:", df_articles.columns)

    # Load test user history
    history_path = data_path / "ebnerd_testset/test/history.parquet"
    df_history = (
        pl.read_parquet(history_path)
        .select([DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL])
        .head(10)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
    )

    # Load behaviors
    behaviors_path = data_path / "ebnerd_testset/test/behaviors.parquet"
    df_behaviors = pl.read_parquet(behaviors_path).head(10)

    # Ensure DEFAULT_CLICKED_ARTICLES_COL exists
    if DEFAULT_CLICKED_ARTICLES_COL not in df_behaviors.columns:
        print("[INFO] Adding missing clicked articles column for test data.")
        df_behaviors = df_behaviors.with_columns(
            pl.Series(DEFAULT_CLICKED_ARTICLES_COL, [[]] * df_behaviors.height)
        )

    # Process behaviors
    df_behaviors = (
        df_behaviors
        .select([DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL])
        .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias("n"))
        .pipe(create_binary_labels_column)
    )

    # Create article-to-token mapping
    article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=token_col_title)

    print("Test data loaded and processed successfully.")
    return df_behaviors, article_mapping, word2vec_embedding



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
    Evaluate the NRMS model on test data.

    Args:
        test_loader: DataLoader for the test set.
        model: Trained NRMS model.
        device: Target device (CPU/GPU).

    Returns:
        Tuple: Test Accuracy and AUC Score
    """
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_idx, ((his_input_title, pred_input_title), labels) in enumerate(test_loader):
            try:
                # Move data to the target device
                his_input_title = his_input_title.to(device)
                pred_input_title = pred_input_title.to(device)
                labels = labels.to(device)

                # Forward pass
                preds, _ = model(his_input_title, pred_input_title)

                # Evaluate predictions
                predicted_labels = preds.argmax(dim=1)
                correct += (predicted_labels == labels).sum().item()
                total += labels.size(0)

                # Store predictions and labels for AUC calculation
                all_preds.extend(preds.softmax(dim=1)[:, 1].cpu().numpy())  # Assuming positive class at index 1
                all_labels.extend(labels.cpu().numpy())

            except Exception as e:
                print(f"[ERROR] Skipping Batch {batch_idx}: {e}")
                continue

    # Calculate evaluation metrics
    accuracy = correct / total if total > 0 else 0
    try:
        auc_score = auc_score_custom(np.array(all_labels), np.array(all_preds))
    except Exception as e:
        auc_score = float('nan')
        print(f"[ERROR] AUC Calculation Failed: {e}")

    print(f"Test Accuracy: {accuracy:.4f}, Test AUC: {auc_score:.4f}")
    return accuracy, auc_score



# def custom_collate_fn_train(batch, history_len):
#     padded_batch = []
#     #print(batch)

#     for (history, prediction), labels in batch:
#         #print(f"Original history shape: {history.shape}")
#         history_padded = torch.zeros((1, history_len, history.size(-1)))
#         history_padded[0, :min(history.size(1), history_len)] = history[0, :history_len]
#         #print(f"Padded history shape: {history_padded.shape}")
#         padded_batch.append(((history_padded, prediction), labels))

#     return default_collate(padded_batch)

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

from torch.utils.data import default_collate
import torch

from torch.utils.data import default_collate
import torch

# def custom_collate_fn(batch):
#     """
#     Custom collate function with strict size validation.
#     Skips samples with mismatched shapes.
#     """
#     valid_samples = []
#     expected_shape = None

#     for (his_input_title, pred_input_title), labels in batch:
#         # Check shape consistency
#         if (
#             his_input_title.ndim == 3 
#             and pred_input_title.ndim == 3 
#             and labels.ndim == 2
#             and his_input_title.shape[1:] == (30, 30)
#             and pred_input_title.shape[1:] == (1, 30)
#         ):


#             # Validate batch dimension consistency
#             current_shape = his_input_title.shape[0]
#             if expected_shape is None:
#                 expected_shape = current_shape
#             if current_shape == expected_shape:
#                 valid_samples.append(((his_input_title, pred_input_title), labels))
#             else:
#                 print(
#                     f"[ERROR] Batch Size Mismatch: Expected {expected_shape}, Got {current_shape}"
#                 )
#         else:
#             print(
#                 f"[ERROR] Skipping Invalid Sample - Shapes: "
#                 f"History {his_input_title.shape}, Prediction {pred_input_title.shape}, Labels {labels.shape}"
#             )

#     if not valid_samples:
#         print("[WARNING] No valid samples in this batch. Skipping...")
#         return float('inf'), 0.0, 0.0  # Return dummy values to avoid breaking


def pad_batch(batch, max_size, target_shape):
    if batch.size(0) > max_size:
        raise ValueError(f"Batch size {batch.size(0)} exceeds maximum allowed size {max_size}")

    if batch.size()[1:] != torch.Size(target_shape):
        raise ValueError(f"Shape mismatch: batch shape {batch.shape}, expected {target_shape}")

    padded_batch = torch.zeros(max_size, *target_shape, device=batch.device)
    padded_batch[:batch.size(0)] = batch
    return padded_batch


def custom_collate_fn(batch):
    fixed_batch_size = 32
    padded_samples = []

    for batch_idx, ((his_input_title, pred_input_title), labels) in enumerate(batch):
        try:
            if his_input_title.size(0) > fixed_batch_size or pred_input_title.size(0) > fixed_batch_size:
                print(f"[WARNING] Skipping Batch {batch_idx}: Batch size exceeds {fixed_batch_size}")
                continue

            his_input_title_padded = pad_batch(his_input_title, fixed_batch_size, (30, 30))
            pred_input_title_padded = pad_batch(pred_input_title.squeeze(1), fixed_batch_size, (30,))
            labels_padded = pad_batch(labels, fixed_batch_size, (1,))

            # Ensure Correct Shape
            assert his_input_title_padded.size() == torch.Size([fixed_batch_size, 30, 30]), \
                f"[ERROR] Incorrect his_input_title_padded shape: {his_input_title_padded.size()}"
            assert pred_input_title_padded.size() == torch.Size([fixed_batch_size, 30]), \
                f"[ERROR] Incorrect pred_input_title_padded shape: {pred_input_title_padded.size()}"

            padded_samples.append(((his_input_title_padded, pred_input_title_padded), labels_padded))

        except ValueError as e:
            print(f"[WARNING] Skipping Batch {batch_idx}: {e}")
            continue

    if not padded_samples:
        raise ValueError("All batches were skipped due to shape mismatches.")

    return default_collate(padded_samples)


# Main Script
if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Paths and constants
    BASE_PATH = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_PATH.joinpath("data").joinpath(dataset_name)
    LOCAL_TOKENIZER_PATH = BASE_PATH.joinpath("data").joinpath("local-tokenizer")
    LOCAL_MODEL_PATH = BASE_PATH.joinpath("data").joinpath("local-tokenizer-model")
    TEST_DATA_PATH = BASE_PATH.joinpath("data")

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
    val_loader = DataLoader(val_dataloader, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=custom_collate_fn)


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
    epochs = 1
    for epoch in range(epochs):
        # # Train the model
        # with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}") as pbar:
        #     train_loss, train_acc, train_auc = train_model(pbar, model, criterion, optimizer, device)
        # print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Train Auc = {train_auc:.4f}")

        # # Validate the model
        # with tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}") as pbar:
        #     val_loss, val_acc, val_auc = validate_model(pbar, model, criterion, device)
        # print(f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}, Val Auc = {val_auc:.4f}")
        pass


    #torch.cuda.empty_cache()
    # Load and evaluate on test data 
    TEST_DATA_PATH = BASE_PATH.joinpath("data")
    LOCAL_TOKENIZER_PATH = BASE_PATH.joinpath("data/local-tokenizer")
    LOCAL_MODEL_PATH = BASE_PATH.joinpath("data/local-tokenizer-model")

    # Call load_test_data with all required arguments
    df_test_behaviors, test_article_mapping, word2vec_embedding = load_test_data(
        TEST_DATA_PATH, title_size=30, history_size=30, 
        tokenizer_path=LOCAL_TOKENIZER_PATH, model_path=LOCAL_MODEL_PATH
    )

    # Evaluate the model after loading test data
    print("Evaluating model on test data...")

    # Initialize DataLoader for test evaluation
    test_dataloader = NRMSDataLoader(
        behaviors=df_test_behaviors,
        article_dict=test_article_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        unknown_representation="zeros",
        eval_mode=True,
        batch_size=1,  # Use batch size 1 for evaluation
    )
    test_loader = DataLoader(test_dataloader, batch_size=None, shuffle=False)

    # Perform evaluation
    test_accuracy, test_auc = evaluate_model(test_loader, model, device)

    # Log results if Neptune is enabled
    if not args.debug:
        run["test/accuracy"].log(test_accuracy)
        run["test/auc"].log(test_auc)

    print("Evaluation Complete.")


if not args.debug:
    run.stop()

print("Done")
