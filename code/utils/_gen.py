import polars as pl

def create_lookup_dict(df: pl.DataFrame, key: str, value: str) -> dict:
    """
    Creates a dictionary lookup table from a Pandas-like DataFrame.

    Args:
        df (pl.DataFrame): The DataFrame from which to create the lookup table.
        key (str): The name of the column containing the keys for the lookup table.
        value (str): The name of the column containing the values for the lookup table.

    Returns:
        dict: A dictionary where the keys are the values from the `key` column of the DataFrame
            and the values are the values from the `value` column of the DataFrame.

    Example:
        >>> df = pl.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        >>> create_lookup_dict(df, 'id', 'name')
            {1: 'Alice', 2: 'Bob', 3: 'Charlie'}
    """
    return dict(zip(df[key], df[value]))