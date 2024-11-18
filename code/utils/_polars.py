import polars as pl
from utils._python import generate_unique_name

def _check_columns_in_df(df: pl.DataFrame, columns: list[str]) -> None:
    """
    Checks whether all specified columns are present in a Polars DataFrame.
    Raises a ValueError if any of the specified columns are not present in the DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        columns (list[str]): The names of the columns to check for.

    Returns:
        None.

    Examples:
    >>> df = pl.DataFrame({"user_id": [1], "first_name": ["J"]})
    >>> check_columns_in_df(df, columns=["user_id", "not_in"])
        ValueError: Invalid input provided. The dataframe does not contain columns ['not_in'].
    """
    columns_not_in_df = [col for col in columns if col not in df.columns]
    if columns_not_in_df:
        raise ValueError(
            f"Invalid input provided. The DataFrame does not contain columns {columns_not_in_df}."
        )


def shuffle_rows(df: pl.DataFrame, seed: int = None) -> pl.DataFrame:
    """
    Shuffle the rows of a DataFrame. This methods allows for LazyFrame,
    whereas, 'df.sample(fraction=1)' is not compatible.

    Examples:
    >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    >>> shuffle_rows(df.lazy(), seed=123).collect()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 1   ┆ 1   │
        │ 3   ┆ 3   ┆ 3   │
        │ 2   ┆ 2   ┆ 2   │
        └─────┴─────┴─────┘
    >>> shuffle_rows(df.lazy(), seed=None).collect().sort("a")
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 1   ┆ 1   │
        │ 2   ┆ 2   ┆ 2   │
        │ 3   ┆ 3   ┆ 3   │
        └─────┴─────┴─────┘

    Test_:
    >>> all([sum(row) == row[0]*3 for row in shuffle_rows(df, seed=None).iter_rows()])
        True

    Note:
        Be aware that 'pl.all().shuffle()' shuffles columns-wise, i.e., with if pl.all().shuffle(None)
        each column's element are shuffled independently from each other (example might change with no seed):
    >>> df_ = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]}).select(pl.all().shuffle(None)).sort("a")
    >>> df_
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 3   ┆ 1   │
        │ 2   ┆ 2   ┆ 3   │
        │ 3   ┆ 1   ┆ 2   │
        └─────┴─────┴─────┘
    >>> all([sum(row) == row[0]*3 for row in shuffle_rows(df_, seed=None).iter_rows()])
        False
    """
    seed = seed if seed is not None else random.randint(1, 1_000_000)
    return df.select(pl.all().shuffle(seed))

def concat_str_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    >>> df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "first_name": ["John", "Jane", "Alice"],
                "last_name": ["Doe", "Doe", "Smith"],
            }
        )
    >>> concatenated_df, concatenated_column_name = concat_str_columns(df, columns=['first_name', 'last_name'])
    >>> concatenated_df
        shape: (3, 4)
        ┌─────┬────────────┬───────────┬──────────────────────┐
        │ id  ┆ first_name ┆ last_name ┆ first_name-last_name │
        │ --- ┆ ---        ┆ ---       ┆ ---                  │
        │ i64 ┆ str        ┆ str       ┆ str                  │
        ╞═════╪════════════╪═══════════╪══════════════════════╡
        │ 1   ┆ John       ┆ Doe       ┆ John Doe             │
        │ 2   ┆ Jane       ┆ Doe       ┆ Jane Doe             │
        │ 3   ┆ Alice      ┆ Smith     ┆ Alice Smith          │
        └─────┴────────────┴───────────┴──────────────────────┘
    """
    concat_name = "-".join(columns)
    concat_columns = df.select(pl.concat_str(columns, separator=" ").alias(concat_name))
    return df.with_columns(concat_columns), concat_name

def shuffle_list_column(
    df: pl.DataFrame, column: str, seed: int = None
) -> pl.DataFrame:
    """Shuffles the values in a list column of a DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to shuffle.
        seed (int, optional): An optional seed value.
            Defaults to None.

    Returns:
        pl.DataFrame: A new DataFrame with the specified column shuffled.

    Example:
    >>> df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "list_col": [["a-", "b-", "c-"], ["a#", "b#"], ["a@", "b@", "c@"]],
                "rdn": ["h", "e", "y"],
            }
        )
    >>> shuffle_list_column(df, 'list_col', seed=1)
        shape: (3, 3)
        ┌─────┬────────────────────┬─────┐
        │ id  ┆ list_col           ┆ rdn │
        │ --- ┆ ---                ┆ --- │
        │ i64 ┆ list[str]          ┆ str │
        ╞═════╪════════════════════╪═════╡
        │ 1   ┆ ["c-", "b-", "a-"] ┆ h   │
        │ 2   ┆ ["a#", "b#"]       ┆ e   │
        │ 3   ┆ ["b@", "c@", "a@"] ┆ y   │
        └─────┴────────────────────┴─────┘

    No seed:
    >>> shuffle_list_column(df, 'list_col', seed=None)
        shape: (3, 3)
        ┌─────┬────────────────────┬─────┐
        │ id  ┆ list_col           ┆ rdn │
        │ --- ┆ ---                ┆ --- │
        │ i64 ┆ list[str]          ┆ str │
        ╞═════╪════════════════════╪═════╡
        │ 1   ┆ ["b-", "a-", "c-"] ┆ h   │
        │ 2   ┆ ["a#", "b#"]       ┆ e   │
        │ 3   ┆ ["a@", "c@", "b@"] ┆ y   │
        └─────┴────────────────────┴─────┘

    Test_:
    >>> assert (
            sorted(shuffle_list_column(df, "list_col", seed=None)["list_col"].to_list()[0])
            == df["list_col"].to_list()[0]
        )

    >>> df = pl.DataFrame({
            'id': [1, 2, 3],
            'list_col': [[6, 7, 8], [-6, -7, -8], [60, 70, 80]],
            'rdn': ['h', 'e', 'y']
        })
    >>> shuffle_list_column(df.lazy(), 'list_col', seed=2).collect()
        shape: (3, 3)
        ┌─────┬──────────────┬─────┐
        │ id  ┆ list_col     ┆ rdn │
        │ --- ┆ ---          ┆ --- │
        │ i64 ┆ list[i64]    ┆ str │
        ╞═════╪══════════════╪═════╡
        │ 1   ┆ [7, 6, 8]    ┆ h   │
        │ 2   ┆ [-8, -7, -6] ┆ e   │
        │ 3   ┆ [60, 80, 70] ┆ y   │
        └─────┴──────────────┴─────┘

    Test_:
    >>> assert (
            sorted(shuffle_list_column(df, "list_col", seed=None)["list_col"].to_list()[0])
            == df["list_col"].to_list()[0]
        )
    """
    _COLUMN_ORDER = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMN_ORDER, "_groupby_id")

    df = df.with_row_count(GROUPBY_ID)
    df_shuffle = (
        df.explode(column)
        .pipe(shuffle_rows, seed=seed)
        .group_by(GROUPBY_ID)
        .agg(column)
    )
    return (
        df.drop(column)
        .join(df_shuffle, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMN_ORDER)
    )

def slice_join_dataframes(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    on: str,
    how: str,
) -> pl.DataFrame:
    """
    Join two dataframes optimized for memory efficiency.
    """
    return pl.concat(
        (
            rows.join(
                df2,
                on=on,
                how=how,
            )
            for rows in df1.iter_slices()
        )
    )