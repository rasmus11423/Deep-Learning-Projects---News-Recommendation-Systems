import numpy as np
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

def create_lookup_objects(
    lookup_dictionary: dict[int, np.array], unknown_representation: str
) -> tuple[dict[int, pl.Series], np.array]:
    """Creates lookup objects for efficient data retrieval.

    This function generates a dictionary of indexes and a matrix from the given lookup dictionary.
    The generated lookup matrix has an additional row based on the specified unknown representation
    which could be either zeros or the mean of the values in the lookup dictionary.

    Args:
        lookup_dictionary (dict[int, np.array]): A dictionary where keys are unique identifiers (int)
            and values are some representations which can be any data type, commonly used for lookup operations.
        unknown_representation (str): Specifies the method to represent unknown entries.
            It can be either 'zeros' to represent unknowns with a row of zeros, or 'mean' to represent
            unknowns with a row of mean values computed from the lookup dictionary.

    Raises:
        ValueError: If the unknown_representation is not either 'zeros' or 'mean',
            a ValueError will be raised.

    Returns:
        tuple[dict[int, pl.Series], np.array]: A tuple containing two items:
            - A dictionary with the same keys as the lookup_dictionary where values are polars Series
                objects containing a single value, which is the index of the key in the lookup dictionary.
            - A numpy array where the rows correspond to the values in the lookup_dictionary and an
                additional row representing unknown entries as specified by the unknown_representation argument.

    Example:
    >>> data = {
            10: np.array([0.1, 0.2, 0.3]),
            20: np.array([0.4, 0.5, 0.6]),
            30: np.array([0.7, 0.8, 0.9]),
        }
    >>> lookup_dict, lookup_matrix = create_lookup_objects(data, "zeros")

    >>> lookup_dict
        {10: shape: (1,)
            Series: '' [i64]
            [
                    1
            ], 20: shape: (1,)
            Series: '' [i64]
            [
                    2
            ], 30: shape: (1,)
            Series: '' [i64]
            [
                    3
        ]}
    >>> lookup_matrix
        array([[0. , 0. , 0. ],
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]])
    """
    # MAKE LOOKUP DICTIONARY
    lookup_indexes = {
        id: pl.Series("", [i]) for i, id in enumerate(lookup_dictionary, start=1)
    }
    # MAKE LOOKUP MATRIX
    lookup_matrix = np.array(list(lookup_dictionary.values()))

    if unknown_representation == "zeros":
        UNKNOWN_ARRAY = np.zeros(lookup_matrix.shape[1], dtype=lookup_matrix.dtype)
    elif unknown_representation == "mean":
        UNKNOWN_ARRAY = np.mean(lookup_matrix, axis=0, dtype=lookup_matrix.dtype)
    else:
        raise ValueError(
            f"'{unknown_representation}' is not a specified method. Can be either 'zeros' or 'mean'."
        )

    lookup_matrix = np.vstack([UNKNOWN_ARRAY, lookup_matrix])
    return lookup_indexes, lookup_matrix