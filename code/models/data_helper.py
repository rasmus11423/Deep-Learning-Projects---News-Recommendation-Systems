from pathlib import Path
import polars as pl
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from models.evaluation import auc_score_custom
import argparse
from models.nrms_pytorch import NRMSModel, UserEncoder, NewsEncoder
from utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_USER_COL,
    DEFAULT_IMPRESSION_ID_COL
)
from utils._behaviors import (
    truncate_history,
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_prediction_scores,
    add_known_user_column
)
from utils._polars import slice_join_dataframes, concat_str_columns
from utils._nlp import get_transformers_word_embeddings
from utils._articles import create_article_id_to_value_mapping, convert_text2encoding_with_transformers

# Define configurations
COLUMNS = [
        DEFAULT_USER_COL,
        DEFAULT_HISTORY_ARTICLE_ID_COL,
        DEFAULT_INVIEW_ARTICLES_COL,
        DEFAULT_CLICKED_ARTICLES_COL,
        DEFAULT_IMPRESSION_ID_COL,
    ]

def grab_data(dataset_name,HISTORY_SIZE):
    def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.LazyFrame:
        """
        Load ebnerd - function
        """
        df_history = (
            pl.scan_parquet(path.joinpath("history.parquet"))
            .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
            .pipe(
                truncate_history,
                column=DEFAULT_HISTORY_ARTICLE_ID_COL,
                history_size=history_size,
                padding_value=0,
                enable_warning=False,
            )
        ).lazy()

        df_behaviors = (
            pl.scan_parquet(path.joinpath("behaviors.parquet"))
            .collect()
            .pipe(
                slice_join_dataframes,
                df2=df_history.collect(),
                on=DEFAULT_USER_COL,
                how="left",
            )
        )
        return df_behaviors

    BASE_PATH = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_PATH.joinpath("data").joinpath(dataset_name)

    TRAIN_PATH = DATA_PATH.joinpath("train")
    VALIDATION_PATH = DATA_PATH.joinpath("validation")

    FRACTION = 0.4

    # Load data
    df_train = (
    ebnerd_from_path(TRAIN_PATH, history_size=HISTORY_SIZE)
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
        ebnerd_from_path(VALIDATION_PATH, history_size=HISTORY_SIZE)
        .select(COLUMNS)
        .pipe(create_binary_labels_column)
        .sample(fraction=FRACTION)
    )

    return df_train,df_validation

def grab_embeded_articles(tokenizer_path,model_path,dataset_name, MAX_TITLE_LENGTH):

    BASE_PATH = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_PATH.joinpath("data").joinpath(dataset_name)

    print("Loading articles and generating embeddings...")
    # Load transformer model and tokenizer
    transformer_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    transformer_model = AutoModel.from_pretrained(model_path)
    
    # Initialize word embeddings

    word2vec_embedding = get_transformers_word_embeddings(transformer_model)

    ARTICLES_PATH = DATA_PATH.joinpath("articles.parquet")
    df_articles = pl.read_parquet(ARTICLES_PATH)

    # Process articles
    TEXT_COLUMNS_TO_USE = ["article_id", "category", "sentiment_label"]

    df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
    df_articles, token_col_title = convert_text2encoding_with_transformers(
        df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
    )

    article_mapping = create_article_id_to_value_mapping(
        df=df_articles, value_col=token_col_title
    )


    return article_mapping, word2vec_embedding


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

def train_model(train_dataloader, model, criterion, optimizer, device,args,run):
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

    train_loss /= len(train_dataloader)
    if not args.debug:
        run["train/loss"].log(train_loss)
        run["train/accuracy"].log(train_acc)
        run["train/auc"].log(train_auc)
    
    print(f"History size in dataloader: {his_input_title.shape}, NPRatio: {pred_input_title.shape}")
    
    return train_loss, train_acc, train_auc


def validate_model(val_dataloader, model, criterion, device, args, run):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Extract batch
            (his_input_title, pred_input_title), labels = batch

            # Send to device
            his_input_title = his_input_title.to(device)
            pred_input_title = pred_input_title.to(device)
            labels = labels.to(device).view(-1)  # Flatten labels to 1D

            # Forward pass
            preds, _ = model(his_input_title, pred_input_title)

            # Compute loss
            loss = criterion(preds.view(-1), labels)

            # Record loss
            val_loss += loss.item()

            # Convert logits to predictions
            pred_labels = torch.round(torch.sigmoid(preds))

            # Update correct and total counts
            val_correct += (pred_labels.view(-1) == labels).sum().item()
            val_total += labels.size(0)

            # Collect predictions and labels for AUC calculation
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Compute metrics
    val_acc = val_correct / val_total if val_total > 0 else 0
    val_loss /= len(val_dataloader) if len(val_dataloader) > 0 else float('nan')

    # Compute AUC
    try:
        val_auc = auc_score_custom(np.array(all_labels), np.array(all_preds))
    except Exception as e:
        val_auc = float('nan')
        print(f"[ERROR] AUC Calculation Failed: {e}")

    if not args.debug:
        run["validation/loss"].log(val_loss)
        run["validation/accuracy"].log(val_acc)
        run["validation/auc"].log(val_auc)

    return val_loss, val_acc, val_auc
