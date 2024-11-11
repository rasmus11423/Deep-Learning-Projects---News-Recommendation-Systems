from pathlib import Path
import numpy as np
import polars as pl
from  utils.constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_CATEGORY_COL,
    DEFAULT_USER_COL,
)

PATH_DATA = Path("code/data")

TOKEN_COL = "tokens"
N_SAMPLES = "n"
BATCH_SIZE = 100

df_articles = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd_demo", "articles.parquet"))
    .select(pl.col(DEFAULT_ARTICLE_ID_COL, DEFAULT_CATEGORY_COL))
    .with_columns(pl.Series(TOKEN_COL, np.random.randint(0, 20, (11777, 10))))
    .collect()
)





print(df_articles.head())