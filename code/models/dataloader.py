import sys
from pathlib import Path

# Add the parent directory of the current file to the Python path
sys.path.append(str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
from dataclasses import dataclass, field
import tensorflow as tf
from utils._gen import create_lookup_objects
from utils._article_behaviors import map_list_article_id_to_value
from utils._constants import (
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_USER_COL
)
from utils._python import (
    repeat_by_list_values_from_matrix,
    create_lookup_objects,
)


class NewsrecDataLoader(tf.keras.utils.Sequence): # We need __init__, because tf Sequence can't resolve TypeError: __init__() got an unexpected keyword argument 'behaviors'

    """
    A DataLoader for news recommendation.
    """

    def __init__(
        self,
        behaviors: pl.DataFrame,
        history_column: str,
        article_dict: dict[int, any],
        unknown_representation: str,
        eval_mode: bool = False,
        batch_size: int = 32,
        inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
        labels_col: str = DEFAULT_LABELS_COL,
        user_col: str = DEFAULT_USER_COL,
        kwargs: dict = None,
    ):
        self.behaviors = behaviors
        self.history_column = history_column
        self.article_dict = article_dict
        self.unknown_representation = unknown_representation
        self.eval_mode = eval_mode
        self.batch_size = batch_size
        self.inview_col = inview_col
        self.labels_col = labels_col
        self.user_col = user_col
        self.kwargs = kwargs or {}
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]
        self.X, self.y = self.load_data()
        self.set_kwargs(self.kwargs)

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self):
        raise ValueError("Function '__getitem__' needs to be implemented.")

    def load_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        X = self.behaviors.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.behaviors[self.labels_col]
        return X, y

    def set_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            setattr(self, key, value)

class NRMSDataLoader(NewsrecDataLoader): # We need __init__, because tf Sequence can't resolve TypeError: __init__() got an unexpected keyword argument 'behaviors'
    def __init__(
        self,
        behaviors: pl.DataFrame,
        history_column: str,
        article_dict: dict[int, any],
        unknown_representation: str,
        eval_mode: bool = False,
        batch_size: int = 32,
        inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
        labels_col: str = DEFAULT_LABELS_COL,
        user_col: str = DEFAULT_USER_COL,
        kwargs: dict = None,
    ):
        super().__init__(
            behaviors=behaviors,
            history_column=history_column,
            article_dict=article_dict,
            unknown_representation=unknown_representation,
            eval_mode=eval_mode,
            batch_size=batch_size,
            inview_col=inview_col,
            labels_col=labels_col,
            user_col=user_col,
            kwargs=kwargs,
        )

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """
        his_input_title:    (samples, history_size, document_dimension)
        pred_input_title:   (samples, npratio, document_dimension)
        batch_y:            (samples, npratio)
        """
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
            self.transform
        )
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        # =>
        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            # =>
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            # =>
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            # =>
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]
        else:
            batch_y = np.array(batch_y.to_list())
            his_input_title = self.lookup_article_matrix[
                batch_X[self.history_column].to_list()
            ]
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].to_list()
            ]
            pred_input_title = np.squeeze(pred_input_title, axis=2)

        his_input_title = np.squeeze(his_input_title, axis=2)
        return (his_input_title, pred_input_title), batch_y