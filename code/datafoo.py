from pathlib import Path
import numpy as np
import polars as pl
from  utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_SENTIMENT_LABEL_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_CATEGORY_COL,
    DEFAULT_USER_COL,
)

from utils._behaviors import create_binary_labels_column

PATH_DATA = Path("code/data")

TOKEN_COL = "tokens"
N_SAMPLES = "n"
BATCH_SIZE = 100

df_articles = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd_demo", "articles.parquet"))
    .select(pl.col(DEFAULT_ARTICLE_ID_COL, DEFAULT_CATEGORY_COL,DEFAULT_SENTIMENT_LABEL_COL))
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

#print(df_articles.head())
#print(df_history.head())
print(df_behaviors.head())