from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import polars as pl

from utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
)

from utils._behaviors import (
    truncate_history,
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_prediction_scores,
    add_known_user_column
)

from utils._polars import slice_join_dataframes, concat_str_columns
from utils._nlp import get_transformers_word_embeddings
from utils._articles import convert_text2encoding_with_transformers,create_article_id_to_value_mapping

from models.dataloader import NRMSDataLoader
from models.model_config import hparams_nrms
from models.nrms import NRMSModel


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

physical_devices = tf.config.list_physical_devices()
print("Available devices:", physical_devices)


def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
    """
    Load ebnerd - function
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
    )

    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes,
            df2=df_history.collect(),
            on=DEFAULT_USER_COL,
            how="left",
        )
    )
    return df_behaviors

PATH = Path("/Users/astridh/Documents/GitHub/Deep-Learning-Projects---News-Recommendation-Systems?fbclid=IwZXh0bgNhZW0CMTEAAR2W1fXivVjSypOEuVjFfg6jZ5IdOeH2OkDWZcbidgezWxAkAp1PnOoHKBA_aem_LPGF8NWJj3P5GF0SIL4g2w/code/data").expanduser()
DATASPLIT = "ebnerd_demo"
DUMP_DIR = PATH.joinpath("dump_artifacts")
DUMP_DIR.mkdir(exist_ok=True, parents=True)

COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]
HISTORY_SIZE = 20
FRACTION = 0.01

df_train = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=True,
        with_replacement=True,
        seed=123,
    )
    .pipe(create_binary_labels_column)
    .sample(fraction=FRACTION)
)
# =>
df_validation = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "validation"), history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(create_binary_labels_column)
    .sample(fraction=FRACTION)
)

print("head")
print(df_train.head(2))


df_articles = pl.read_parquet(PATH.joinpath(DATASPLIT+"/articles.parquet"))
print(df_articles.head(2))

TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
MAX_TITLE_LENGTH = 30

# LOAD HUGGINGFACE:
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

# We'll init the word embeddings using the
word2vec_embedding = get_transformers_word_embeddings(transformer_model)
#
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
# =>
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=token_col_title
)

train_dataloader = NRMSDataLoader(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=64,
)
val_dataloader = NRMSDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=32,
)

MODEL_NAME = "NRMS"
LOG_DIR = f"downloads/runs/{MODEL_NAME}"
MODEL_WEIGHTS = f"downloads/data/state_dict/{MODEL_NAME}/weights"


hparams_nrms.history_size = HISTORY_SIZE
model = NRMSModel(
    hparams=hparams_nrms,
    word2vec_embedding=word2vec_embedding,
    seed=42,
)
hist = model.model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=1
)
_ = model.model.load_weights(filepath=MODEL_WEIGHTS)

pred_validation = model.scorer.predict(val_dataloader)

df_validation = add_prediction_scores(df_validation, pred_validation.tolist()).pipe(
    add_known_user_column, known_users=df_train[DEFAULT_USER_COL]
)
print(df_validation.head(2))