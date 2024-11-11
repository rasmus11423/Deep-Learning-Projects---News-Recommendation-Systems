import polars as pl
from utils._gen import create_lookup_dict
from utils._constants import DEFAULT_ARTICLE_ID_COL


def create_article_id_to_value_mapping(
    df: pl.DataFrame,
    value_col: str,
    article_col: str = DEFAULT_ARTICLE_ID_COL,
):
    return create_lookup_dict(
        df.select(article_col, value_col), key=article_col, value=value_col
    )