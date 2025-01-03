import polars as pl
from utils._gen import create_lookup_dict
from utils._constants import DEFAULT_ARTICLE_ID_COL
from transformers import AutoTokenizer




def convert_text2encoding_with_transformers(
    df: pl.DataFrame,
    tokenizer: AutoTokenizer,
    column: str,
    max_length: int = None,
) -> pl.DataFrame:
    """
    From EBNerd-Benchmark.
    
    Converts text in a specified DataFrame column to tokens using a provided tokenizer.
    Args:
        df (pl.DataFrame): The input DataFrame containing the text column.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the text. (from transformers import AutoTokenizer)
        column (str): The name of the column containing the text.
        max_length (int, optional): The maximum length of the encoded tokens. Defaults to None.
    Returns:
        pl.DataFrame: A new DataFrame with an additional column containing the encoded tokens.
    Example:
    >>> from transformers import AutoTokenizer
    >>> import polars as pl
    >>> df = pl.DataFrame({
            'text': ['This is a test.', 'Another test string.', 'Yet another one.']
        })
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> encoded_df, new_column = convert_text2encoding_with_transformers(df, tokenizer, 'text', max_length=20)
    >>> print(encoded_df)
        shape: (3, 2)
        ┌──────────────────────┬───────────────────────────────┐
        │ text                 ┆ text_encode_bert-base-uncased │
        │ ---                  ┆ ---                           │
        │ str                  ┆ list[i64]                     │
        ╞══════════════════════╪═══════════════════════════════╡
        │ This is a test.      ┆ [2023, 2003, … 0]             │
        │ Another test string. ┆ [2178, 3231, … 0]             │
        │ Yet another one.     ┆ [2664, 2178, … 0]             │
        └──────────────────────┴───────────────────────────────┘
    >>> print(new_column)
        text_encode_bert-base-uncased
    """
    text = df[column].to_list()
    # set columns
    new_column = f"{column}_encode_{tokenizer.name_or_path}"
    # If 'max_length' is provided then set it, else encode each string its original length
    padding = "max_length" if max_length else False
    encoded_tokens = tokenizer(
        text,
        add_special_tokens=False,
        padding=padding,
        max_length=max_length,
        truncation=True,
    )["input_ids"]
    return df.with_columns(pl.Series(new_column, encoded_tokens)), new_column

def create_article_id_to_value_mapping(
    df: pl.DataFrame,
    value_col: str,
    article_col: str = DEFAULT_ARTICLE_ID_COL,
):
    return create_lookup_dict(
        df.select(article_col, value_col), key=article_col, value=value_col
    )