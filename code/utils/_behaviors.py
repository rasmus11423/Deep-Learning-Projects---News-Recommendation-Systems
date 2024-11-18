import polars as pl
from typing import Any, Iterable
import warnings
import inspect

from utils._constants import (
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_USER_COL,
    DEFAULT_KNOWN_USER_COL
)
from utils._polars import _check_columns_in_df, shuffle_list_column
from utils._gen import create_lookup_dict

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
    _check_columns_in_df(df, [inview_col, clicked_col])
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

def create_user_id_to_int_mapping(
    df: pl.DataFrame, user_col: str = DEFAULT_USER_COL, value_str: str = "id"
):
    return create_lookup_dict(
        df.select(pl.col(user_col).unique()).with_row_index(value_str),
        key=user_col,
        value=value_str,
    )

def truncate_history(
    df: pl.DataFrame,
    column: str,
    history_size: int,
    padding_value: Any = None,
    enable_warning: bool = True,
) -> pl.DataFrame:
    """Truncates the history of a column containing a list of items.

    It is the tail of the values, i.e. the history ids should ascending order
    because each subsequent element (original timestamp) is greater than the previous element

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to truncate.
        history_size (int): The maximum size of the history to retain.
        padding_value (Any): Pad each list with specified value, ensuring
            equal length to each element. Default is None (no padding).
        enable_warning (bool): warn the user that history is expected in ascedings order.
            Default is True

    Returns:
        pl.DataFrame: A new DataFrame with the specified column truncated.

    Examples:
    >>> df = pl.DataFrame(
            {"id": [1, 2, 3], "history": [["a", "b", "c"], ["d", "e", "f", "g"], ["h", "i"]]}
        )
    >>> df
        shape: (3, 2)
        ┌─────┬───────────────────┐
        │ id  ┆ history           │
        │ --- ┆ ---               │
        │ i64 ┆ list[str]         │
        ╞═════╪═══════════════════╡
        │ 1   ┆ ["a", "b", "c"]   │
        │ 2   ┆ ["d", "e", … "g"] │
        │ 3   ┆ ["h", "i"]        │
        └─────┴───────────────────┘
    >>> truncate_history(df, 'history', 3)
        shape: (3, 2)
        ┌─────┬─────────────────┐
        │ id  ┆ history         │
        │ --- ┆ ---             │
        │ i64 ┆ list[str]       │
        ╞═════╪═════════════════╡
        │ 1   ┆ ["a", "b", "c"] │
        │ 2   ┆ ["e", "f", "g"] │
        │ 3   ┆ ["h", "i"]      │
        └─────┴─────────────────┘
    >>> truncate_history(df.lazy(), 'history', 3, '-').collect()
        shape: (3, 2)
        ┌─────┬─────────────────┐
        │ id  ┆ history         │
        │ --- ┆ ---             │
        │ i64 ┆ list[str]       │
        ╞═════╪═════════════════╡
        │ 1   ┆ ["a", "b", "c"] │
        │ 2   ┆ ["e", "f", "g"] │
        │ 3   ┆ ["-", "h", "i"] │
        └─────┴─────────────────┘
    """
    if enable_warning:
        function_name = inspect.currentframe().f_code.co_name
        warnings.warn(f"{function_name}: The history IDs expeced in ascending order")
    if padding_value is not None:
        df = df.with_columns(
            pl.col(column)
            .list.reverse()
            .list.eval(pl.element().extend_constant(padding_value, n=history_size))
            .list.reverse()
        )
    return df.with_columns(pl.col(column).list.tail(history_size))

def remove_positives_from_inview(
    df: pl.DataFrame,
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
    clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
):
    """Removes all positive article IDs from a DataFrame column containing inview articles and another column containing
    clicked articles. Only negative article IDs (i.e., those that appear in the inview articles column but not in the
    clicked articles column) are retained.

    Args:
        df (pl.DataFrame): A DataFrame with columns containing inview articles and clicked articles.

    Returns:
        pl.DataFrame: A new DataFrame with only negative article IDs retained.

    Examples:
    >>> from ebrec.utils._constants import DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "user_id": [1, 1, 2],
                DEFAULT_CLICKED_ARTICLES_COL: [
                    [1, 2],
                    [1],
                    [3],
                ],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                ],
            }
        )
    >>> remove_positives_from_inview(df)
        shape: (3, 3)
        ┌─────────┬─────────────────────┬────────────────────┐
        │ user_id ┆ article_ids_clicked ┆ article_ids_inview │
        │ ---     ┆ ---                 ┆ ---                │
        │ i64     ┆ list[i64]           ┆ list[i64]          │
        ╞═════════╪═════════════════════╪════════════════════╡
        │ 1       ┆ [1, 2]              ┆ [3]                │
        │ 1       ┆ [1]                 ┆ [2, 3]             │
        │ 2       ┆ [3]                 ┆ [1, 2]             │
        └─────────┴─────────────────────┴────────────────────┘
    """
    _check_columns_in_df(df, [inview_col, clicked_col])
    negative_article_ids = (
        list(filter(lambda x: x not in clicked, inview))
        for inview, clicked in zip(df[inview_col].to_list(), df[clicked_col].to_list())
    )
    return df.with_columns(pl.Series(inview_col, list(negative_article_ids)))

def sample_article_ids(
    df: pl.DataFrame,
    n: int,
    with_replacement: bool = False,
    seed: int = None,
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
) -> pl.DataFrame:
    """
    Randomly sample article IDs from each row of a DataFrame with or without replacement

    Args:
        df: A polars DataFrame containing the column of article IDs to be sampled.
        n: The number of article IDs to sample from each list.
        with_replacement: A boolean indicating whether to sample with replacement.
            Default is False.
        seed: An optional seed to use for the random number generator.

    Returns:
        A new polars DataFrame with the same columns as `df`, but with the article
        IDs in the specified column replaced by a list of `n` sampled article IDs.

    Examples:
    >>> from ebrec.utils._constants import DEFAULT_INVIEW_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "clicked": [
                    [1],
                    [4, 5],
                    [7, 8, 9],
                ],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    ["A", "B", "C"],
                    ["D", "E", "F"],
                    ["G", "H", "I"],
                ],
                "col" : [
                    ["h"],
                    ["e"],
                    ["y"]
                ]
            }
        )
    >>> print(df)
        shape: (3, 3)
        ┌──────────────────┬─────────────────┬───────────┐
        │ list_destination ┆ article_ids     ┆ col       │
        │ ---              ┆ ---             ┆ ---       │
        │ list[i64]        ┆ list[str]       ┆ list[str] │
        ╞══════════════════╪═════════════════╪═══════════╡
        │ [1]              ┆ ["A", "B", "C"] ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "E", "F"] ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "H", "I"] ┆ ["y"]     │
        └──────────────────┴─────────────────┴───────────┘
    >>> sample_article_ids(df, n=2, seed=42)
        shape: (3, 3)
        ┌──────────────────┬─────────────┬───────────┐
        │ list_destination ┆ article_ids ┆ col       │
        │ ---              ┆ ---         ┆ ---       │
        │ list[i64]        ┆ list[str]   ┆ list[str] │
        ╞══════════════════╪═════════════╪═══════════╡
        │ [1]              ┆ ["A", "C"]  ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "F"]  ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "I"]  ┆ ["y"]     │
        └──────────────────┴─────────────┴───────────┘
    >>> sample_article_ids(df.lazy(), n=4, with_replacement=True, seed=42).collect()
        shape: (3, 3)
        ┌──────────────────┬───────────────────┬───────────┐
        │ list_destination ┆ article_ids       ┆ col       │
        │ ---              ┆ ---               ┆ ---       │
        │ list[i64]        ┆ list[str]         ┆ list[str] │
        ╞══════════════════╪═══════════════════╪═══════════╡
        │ [1]              ┆ ["A", "A", … "C"] ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "D", … "F"] ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "G", … "I"] ┆ ["y"]     │
        └──────────────────┴───────────────────┴───────────┘
    """
    _check_columns_in_df(df, [inview_col])
    _COLUMNS = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMNS, "_groupby_id")
    df = df.with_row_count(name=GROUPBY_ID)

    df_ = (
        df.explode(inview_col)
        .group_by(GROUPBY_ID)
        .agg(
            pl.col(inview_col).sample(n=n, with_replacement=with_replacement, seed=seed)
        )
    )
    return (
        df.drop(inview_col)
        .join(df_, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMNS)
    )


def sampling_strategy_wu2019(
    df: pl.DataFrame,
    npratio: int,
    shuffle: bool = False,
    with_replacement: bool = True,
    seed: int = None,
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
    clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
) -> pl.DataFrame:
    """
    Samples negative articles from the inview article pool for a given negative-position-ratio (npratio).
    The npratio (negative article per positive article) is defined as the number of negative article samples
    to draw for each positive article sample.

    This function follows the sampling strategy introduced in the paper "NPA: Neural News Recommendation with
    Personalized Attention" by Wu et al. (KDD '19).

    This is done according to the following steps:
    1. Remove the positive click-article id pairs from the DataFrame.
    2. Explode the DataFrame based on the clicked articles column.
    3. Downsample the inview negative article ids for each exploded row using the specified npratio, either
        with or without replacement.
    4. Concatenate the clicked articles back to the inview articles as lists.
    5. Convert clicked articles column to type List(Int)

    References:
        Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. 2019.
        Npa: Neural news recommendation with personalized attention. In KDD, pages 2576-2584. ACM.

    Args:
        df (pl.DataFrame): The input DataFrame containing click-article id pairs.
        npratio (int): The ratio of negative in-view article ids to positive click-article ids.
        shuffle (bool, optional): Whether to shuffle the order of the in-view article ids in each list. Default is True.
        with_replacement (bool, optional): Whether to sample the inview article ids with or without replacement.
            Default is True.
        seed (int, optional): Random seed for reproducibility. Default is None.
        inview_col (int, optional): inview column name. Default is DEFAULT_INVIEW_ARTICLES_COL,
        clicked_col (int, optional): clicked column name. Default is DEFAULT_CLICKED_ARTICLES_COL,

    Returns:
        pl.DataFrame: A new DataFrame with downsampled in-view article ids for each click according to the specified npratio.
        The DataFrame has the same columns as the input DataFrame.

    Raises:
        ValueError: If npratio is less than 0.
        ValueError: If the input DataFrame does not contain the necessary columns.

    Examples:
    >>> from ebrec.utils._constants import DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_INVIEW_ARTICLES_COL
    >>> import polars as pl
    >>> df = pl.DataFrame(
            {
                "impression_id": [0, 1, 2, 3],
                "user_id": [1, 1, 2, 3],
                DEFAULT_INVIEW_ARTICLES_COL: [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3], [1]],
                DEFAULT_CLICKED_ARTICLES_COL: [[1, 2], [1, 3], [1], [1]],
            }
        )
    >>> df
        shape: (4, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ list[i64]           │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 0             ┆ 1       ┆ [1, 2, 3]          ┆ [1, 2]              │
        │ 1             ┆ 1       ┆ [1, 2, … 4]        ┆ [1, 3]              │
        │ 2             ┆ 2       ┆ [1, 2, 3]          ┆ [1]                 │
        │ 3             ┆ 3       ┆ [1]                ┆ [1]                 │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    >>> sampling_strategy_wu2019(df, npratio=1, shuffle=False, with_replacement=True, seed=123)
        shape: (6, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ list[i64]           │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 0             ┆ 1       ┆ [3, 1]             ┆ [1]                 │
        │ 0             ┆ 1       ┆ [3, 2]             ┆ [2]                 │
        │ 1             ┆ 1       ┆ [4, 1]             ┆ [1]                 │
        │ 1             ┆ 1       ┆ [4, 3]             ┆ [3]                 │
        │ 2             ┆ 2       ┆ [3, 1]             ┆ [1]                 │
        │ 3             ┆ 3       ┆ [null, 1]          ┆ [1]                 │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    >>> sampling_strategy_wu2019(df, npratio=1, shuffle=True, with_replacement=True, seed=123)
        shape: (6, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ list[i64]           │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 0             ┆ 1       ┆ [3, 1]             ┆ [1]                 │
        │ 0             ┆ 1       ┆ [2, 3]             ┆ [2]                 │
        │ 1             ┆ 1       ┆ [4, 1]             ┆ [1]                 │
        │ 1             ┆ 1       ┆ [4, 3]             ┆ [3]                 │
        │ 2             ┆ 2       ┆ [3, 1]             ┆ [1]                 │
        │ 3             ┆ 3       ┆ [null, 1]          ┆ [1]                 │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    >>> sampling_strategy_wu2019(df, npratio=2, shuffle=False, with_replacement=True, seed=123)
        shape: (6, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ list[i64]           │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 0             ┆ 1       ┆ [3, 3, 1]          ┆ [1]                 │
        │ 0             ┆ 1       ┆ [3, 3, 2]          ┆ [2]                 │
        │ 1             ┆ 1       ┆ [4, 2, 1]          ┆ [1]                 │
        │ 1             ┆ 1       ┆ [4, 2, 3]          ┆ [3]                 │
        │ 2             ┆ 2       ┆ [3, 2, 1]          ┆ [1]                 │
        │ 3             ┆ 3       ┆ [null, null, 1]    ┆ [1]                 │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    # If we use without replacement, we need to ensure there are enough negative samples:
    >>> sampling_strategy_wu2019(df, npratio=2, shuffle=False, with_replacement=False, seed=123)
        polars.exceptions.ShapeError: cannot take a larger sample than the total population when `with_replacement=false`
    ## Either you'll have to remove the samples or split the dataframe yourself and only upsample the samples that doesn't have enough
    >>> min_neg = 2
    >>> sampling_strategy_wu2019(
            df.filter(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len() > (min_neg + 1)),
            npratio=min_neg,
            shuffle=False,
            with_replacement=False,
            seed=123,
        )
        shape: (2, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ i64                 │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 1             ┆ 1       ┆ [2, 4, 1]          ┆ 1                   │
        │ 1             ┆ 1       ┆ [2, 4, 3]          ┆ 3                   │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    """
    df = (
        # Step 1: Remove the positive 'article_id' from inview articles
        df.pipe(
            remove_positives_from_inview, inview_col=inview_col, clicked_col=clicked_col
        )
        # Step 2: Explode the DataFrame based on the clicked articles column
        .explode(clicked_col)
        # Step 3: Downsample the inview negative 'article_id' according to npratio (negative 'article_id' per positive 'article_id')
        .pipe(
            sample_article_ids,
            n=npratio,
            with_replacement=with_replacement,
            seed=seed,
            inview_col=inview_col,
        )
        # Step 4: Concatenate the clicked articles back to the inview articles as lists
        .with_columns(pl.concat_list([inview_col, clicked_col]))
        # Step 5: Convert clicked articles column to type List(Int):
        .with_columns(pl.col(inview_col).list.tail(1).alias(clicked_col))
    )
    if shuffle:
        df = shuffle_list_column(df, inview_col, seed)
    return df

def add_known_user_column(
    df: pl.DataFrame,
    known_users: Iterable[int],
    user_col: str = DEFAULT_USER_COL,
    known_user_col: str = DEFAULT_KNOWN_USER_COL,
) -> pl.DataFrame:
    """
    Adds a new column to the DataFrame indicating whether the user ID is in the list of known users.
    Args:
        df: A Polars DataFrame object.
        known_users: An iterable of integers representing the known user IDs.
    Returns:
        A new Polars DataFrame with an additional column 'is_known_user' containing a boolean value
        indicating whether the user ID is in the list of known users.
    Examples:
        >>> df = pl.DataFrame({'user_id': [1, 2, 3, 4]})
        >>> add_known_user_column(df, [2, 4])
            shape: (4, 2)
            ┌─────────┬───────────────┐
            │ user_id ┆ is_known_user │
            │ ---     ┆ ---           │
            │ i64     ┆ bool          │
            ╞═════════╪═══════════════╡
            │ 1       ┆ false         │
            │ 2       ┆ true          │
            │ 3       ┆ false         │
            │ 4       ┆ true          │
            └─────────┴───────────────┘
    """
    return df.with_columns(pl.col(user_col).is_in(known_users).alias(known_user_col))



def add_prediction_scores(
    df: pl.DataFrame,
    scores: Iterable[float],
    prediction_scores_col: str = "scores",
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
) -> pl.DataFrame:
    """
    Adds prediction scores to a DataFrame for the corresponding test predictions.

    Args:
        df (pl.DataFrame): The DataFrame to which the prediction scores will be added.
        test_prediction (Iterable[float]): A list, array or simialr of prediction scores for the test data.

    Returns:
        pl.DataFrame: The DataFrame with the prediction scores added.

    Raises:
        ValueError: If there is a mismatch in the lengths of the list columns.

    >>> from ebrec.utils._constants import DEFAULT_INVIEW_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "id": [1,2],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    [1, 2, 3],
                    [4, 5],
                ],
            }
        )
    >>> test_prediction = [[0.3], [0.4], [0.5], [0.6], [0.7]]
    >>> add_prediction_scores(df.lazy(), test_prediction).collect()
        shape: (2, 3)
        ┌─────┬─────────────┬────────────────────────┐
        │ id  ┆ article_ids ┆ prediction_scores_test │
        │ --- ┆ ---         ┆ ---                    │
        │ i64 ┆ list[i64]   ┆ list[f32]              │
        ╞═════╪═════════════╪════════════════════════╡
        │ 1   ┆ [1, 2, 3]   ┆ [0.3, 0.4, 0.5]        │
        │ 2   ┆ [4, 5]      ┆ [0.6, 0.7]             │
        └─────┴─────────────┴────────────────────────┘
    ## The input can can also be an np.array
    >>> add_prediction_scores(df.lazy(), np.array(test_prediction)).collect()
        shape: (2, 3)
        ┌─────┬─────────────┬────────────────────────┐
        │ id  ┆ article_ids ┆ prediction_scores_test │
        │ --- ┆ ---         ┆ ---                    │
        │ i64 ┆ list[i64]   ┆ list[f32]              │
        ╞═════╪═════════════╪════════════════════════╡
        │ 1   ┆ [1, 2, 3]   ┆ [0.3, 0.4, 0.5]        │
        │ 2   ┆ [4, 5]      ┆ [0.6, 0.7]             │
        └─────┴─────────────┴────────────────────────┘
    """
    GROUPBY_ID = generate_unique_name(df.columns, "_groupby_id")
    # df_preds = pl.DataFrame()
    scores = (
        df.lazy()
        .select(pl.col(inview_col))
        .with_row_index(GROUPBY_ID)
        .explode(inview_col)
        .with_columns(pl.Series(prediction_scores_col, scores).explode())
        .group_by(GROUPBY_ID)
        .agg(inview_col, prediction_scores_col)
        .sort(GROUPBY_ID)
        .collect()
    )
    return df.with_columns(scores.select(prediction_scores_col))