from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_SENTIMENT_LABEL_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_CATEGORY_COL,
    DEFAULT_USER_COL,
)
from utils._behaviors import (
    create_binary_labels_column,
    create_user_id_to_int_mapping
)
from utils._articles import create_article_id_to_value_mapping
from models.dataloader_pytorch import NRMSDataLoader  # Ensure PyTorch version is imported

PATH_DATA = Path("data/")

TOKEN_COL = "tokens"
N_SAMPLES = "n"
BATCH_SIZE = 100

df_articles = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd_demo", "articles.parquet"))
    .select(pl.col(DEFAULT_ARTICLE_ID_COL, DEFAULT_CATEGORY_COL, DEFAULT_SENTIMENT_LABEL_COL))
    .with_columns(pl.Series(TOKEN_COL, np.random.randint(0, 20, (11777, 10))))
    .collect()
)

df_history = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd_demo/train", "history.parquet"))
    .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
    .with_columns(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(3))
)

df_behaviors = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd_demo/train", "behaviors.parquet"))
    .select(DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL)
    .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias(N_SAMPLES))
    .join(df_history, on=DEFAULT_USER_COL, how="left")
    .collect()
    .pipe(create_binary_labels_column)
)

# => MAPPINGS:
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=TOKEN_COL
)

user_mapping = create_user_id_to_int_mapping(df=df_behaviors)
# => NPRATIO IMPRESSION - SAME LENGTHS:
df_behaviors_train = df_behaviors.filter(pl.col(N_SAMPLES) == pl.col(N_SAMPLES).min())
# => FOR TEST-DATALOADER
label_lengths = df_behaviors[DEFAULT_INVIEW_ARTICLES_COL].list.len().to_list()

def test_NRMSDataLoader():
    train_dataloader = NRMSDataLoader(
        behaviors=df_behaviors_train,
        article_dict=article_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        unknown_representation="zeros",
        eval_mode=False,
        batch_size=BATCH_SIZE,
    )

    # Wrap with PyTorch DataLoader
    data_loader = DataLoader(train_dataloader, batch_size=None, shuffle=False)

    print("NRMSDataLoader initialized successfully.")

    # Fetch first batch
    batch = next(iter(data_loader))

    # Iterate over batches
    for batch in data_loader:
        print()
        print("Train Dataloader")
        print("Batch inputs shape (his_input_title):", batch[0][0].shape)
        print("Batch inputs shape (pred_input_title):", batch[0][1].shape)
        print("Batch labels shape:", batch[1].shape)
        break  # Exit after the first batch for testing purposes

    assert len(train_dataloader) == int(np.ceil(df_behaviors_train.shape[0]))
    assert len(batch) == 2, "There should be two outputs: (inputs, labels)"
    assert len(batch[0]) == 2, "NRMS has two outputs (his_input_title, pred_input_title_one)"

    for type_in_batch in batch[0][0]:
        assert isinstance(
            type_in_batch.ravel()[0], torch.Tensor
        ), "Expected output to be tensor; used for lookup value"

    assert isinstance(
        batch[1].ravel()[0], torch.Tensor
    ), "Expected output to be tensor; this is label"

    test_dataloader = NRMSDataLoader(
        behaviors=df_behaviors,
        article_dict=article_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        unknown_representation="zeros",
        eval_mode=True,
        batch_size=BATCH_SIZE,
    )

    test_loader = DataLoader(test_dataloader, batch_size=None, shuffle=False)

    batch = next(iter(test_loader))
    print()
    print("Test Dataloader")
    print("Batch inputs shape (his_input_title):", batch[0][0].shape)
    print("Batch inputs shape (pred_input_title):", batch[0][1].shape)
    print("Batch labels shape:", batch[1].shape)

test_NRMSDataLoader()
