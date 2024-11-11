import polars as pl
from utils._constants import (
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL
)
from utils._polars import check_columns_in_df

def generate_unique_name(existing_names: list[str], base_name: str = "new_name"):
    """
    Generate a unique name based on a list of existing names.

    Args:
        existing_names (list of str): The list of existing names.
        base_name (str): The base name to start with. Default is 'newName'.

    Returns:
        str: A unique name.
    Example
    >>> existing_names = ['name1', 'name2', 'newName', 'newName_1']
    >>> generate_unique_name(existing_names, 'newName')
        'newName_2'
    """
    if base_name not in existing_names:
        return base_name

    suffix = 1
    new_name = f"{base_name}_{suffix}"

    while new_name in existing_names:
        suffix += 1
        new_name = f"{base_name}_{suffix}"

    return new_name


def create_binary_labels_column(
    df: pl.DataFrame,
    shuffle: bool = True,
    seed: int = None,
    clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
    label_col: str = DEFAULT_LABELS_COL,
) -> pl.DataFrame:
    """Creates a new column in a DataFrame containing binary labels indicating
    whether each article ID in the "article_ids" column is present in the corresponding
    "list_destination" column.

    Args:
        df (pl.DataFrame): The input DataFrame.

    Returns:
        pl.DataFrame: A new DataFrame with an additional "labels" column.

    Examples:
    >>> from ebrec.utils._constants import (
            DEFAULT_CLICKED_ARTICLES_COL,
            DEFAULT_INVIEW_ARTICLES_COL,
            DEFAULT_LABELS_COL,
        )
    >>> df = pl.DataFrame(
            {
                DEFAULT_INVIEW_ARTICLES_COL: [[1, 2, 3], [4, 5, 6], [7, 8]],
                DEFAULT_CLICKED_ARTICLES_COL: [[2, 3, 4], [3, 5], None],
            }
        )
    >>> create_binary_labels_column(df)
        shape: (3, 3)
        ┌────────────────────┬─────────────────────┬───────────┐
        │ article_ids_inview ┆ article_ids_clicked ┆ labels    │
        │ ---                ┆ ---                 ┆ ---       │
        │ list[i64]          ┆ list[i64]           ┆ list[i8]  │
        ╞════════════════════╪═════════════════════╪═══════════╡
        │ [1, 2, 3]          ┆ [2, 3, 4]           ┆ [0, 1, 1] │
        │ [4, 5, 6]          ┆ [3, 5]              ┆ [0, 1, 0] │
        │ [7, 8]             ┆ null                ┆ [0, 0]    │
        └────────────────────┴─────────────────────┴───────────┘
    >>> create_binary_labels_column(df.lazy(), shuffle=True, seed=123).collect()
        shape: (3, 3)
        ┌────────────────────┬─────────────────────┬───────────┐
        │ article_ids_inview ┆ article_ids_clicked ┆ labels    │
        │ ---                ┆ ---                 ┆ ---       │
        │ list[i64]          ┆ list[i64]           ┆ list[i8]  │
        ╞════════════════════╪═════════════════════╪═══════════╡
        │ [3, 1, 2]          ┆ [2, 3, 4]           ┆ [1, 0, 1] │
        │ [5, 6, 4]          ┆ [3, 5]              ┆ [1, 0, 0] │
        │ [7, 8]             ┆ null                ┆ [0, 0]    │
        └────────────────────┴─────────────────────┴───────────┘
    Test_:
    >>> assert create_binary_labels_column(df, shuffle=False)[DEFAULT_LABELS_COL].to_list() == [
            [0, 1, 1],
            [0, 1, 0],
            [0, 0],
        ]
    >>> assert create_binary_labels_column(df, shuffle=True)[DEFAULT_LABELS_COL].list.sum().to_list() == [
            2,
            1,
            0,
        ]
    """
    check_columns_in_df(df, [inview_col, clicked_col])
    _COLUMNS = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMNS, "_groupby_id")

    df = df.with_row_index(GROUPBY_ID)

    # if shuffle:
    #     df = shuffle_list_column(df, column=inview_col, seed=seed)

    df_labels = (
        df.explode(inview_col)
        .with_columns(
            pl.col(inview_col).is_in(pl.col(clicked_col)).cast(pl.Int8).alias(label_col)
        )
        .group_by(GROUPBY_ID)
        .agg(label_col)
    )
    return (
        df.join(df_labels, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMNS + [label_col])
    )