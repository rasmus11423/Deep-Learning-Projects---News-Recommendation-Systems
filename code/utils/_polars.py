import polars as pl

def check_columns_in_df(df: pl.DataFrame, columns: list[str]) -> None:
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