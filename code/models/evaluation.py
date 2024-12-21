import numpy as np

def auc_score_custom(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Area Under the Curve (AUC) score for the Receiver Operating Characteristic (ROC) curve using a
    custom method. This implementation is particularly useful for understanding basic ROC curve properties and
    for educational purposes to demonstrate how AUC scores can be manually calculated.

    This function may produce slightly different results compared to standard library implementations (e.g., sklearn's roc_auc_score)
    in cases where positive and negative predictions have the same score. The function treats the problem as a binary classification task,
    comparing the prediction scores for positive instances against those for negative instances directly.

    Args:
        y_true (np.ndarray): A binary array indicating the true classification (1 for positive class and 0 for negative class).
        y_pred (np.ndarray): An array of scores as predicted by a model, indicating the likelihood of each instance being positive.

    Returns:
        float: The calculated AUC score, representing the probability that a randomly chosen positive instance is ranked
                higher than a randomly chosen negative instance based on the prediction scores.

    Raises:
        ValueError: If `y_true` and `y_pred` do not have the same length or if they contain invalid data types.

    Examples:
        >>> y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        >>> y_pred = np.array([0.9999, 0.9838, 0.5747, 0.8485, 0.8624, 0.4502, 0.3357, 0.8985])
        >>> auc_score_custom(y_true, y_pred)
            0.9333333333333333
        >>> from sklearn.metrics import roc_auc_score
        >>> roc_auc_score(y_true, y_pred)
            0.9333333333333333

        An error will occur when pos/neg prediction have same score:
        >>> y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        >>> y_pred = np.array([0.9999, 0.8, 0.8, 0.8485, 0.8624, 0.4502, 0.3357, 0.8985])
        >>> auc_score_custom(y_true, y_pred)
            0.7333
        >>> roc_auc_score(y_true, y_pred)
            0.7667
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true_bool = y_true.astype(np.bool_)
    # Index:
    pos_scores = y_pred[y_true_bool]
    neg_scores = y_pred[np.logical_not(y_true_bool)]
    # Arrange:
    pos_scores = np.repeat(pos_scores, len(neg_scores))
    neg_scores = np.tile(neg_scores, sum(y_true_bool))
    assert len(neg_scores) == len(pos_scores)
    return (pos_scores > neg_scores).sum() / len(neg_scores)


from itertools import compress
from typing import Iterable
import numpy as np
import json

from evaluation.utils import convert_to_binary
from evaluation.protocols import Metric

from evaluation.metrics import (
    mean_squared_error,
    accuracy_score,
    roc_auc_score,
    ndcg_score,
    mrr_score,
    log_loss,
    f1_score,
)


class AccuracyScore(Metric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = "accuracy"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                accuracy_score(
                    each_labels, convert_to_binary(each_preds, self.threshold)
                )
                for each_labels, each_preds in zip(y_true, y_pred)
            ]
        )
        return float(res)


class F1Score(Metric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = "f1"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                f1_score(each_labels, convert_to_binary(each_preds, self.threshold))
                for each_labels, each_preds in zip(y_true, y_pred)
            ]
        )
        return float(res)


class RootMeanSquaredError(Metric):
    def __init__(self):
        self.name = "rmse"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                np.sqrt(mean_squared_error(each_labels, each_preds))
                for each_labels, each_preds in zip(y_true, y_pred)
            ]
        )
        return float(res)


class AucScore(Metric):
    def __init__(self):
        self.name = "auc"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                roc_auc_score(each_labels, each_preds)
                for each_labels, each_preds in zip(y_true, y_pred)
            ]
        )
        return float(res)


class LogLossScore(Metric):
    def __init__(self):
        self.name = "logloss"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                log_loss(
                    each_labels,
                    [max(min(p, 1.0 - 10e-12), 10e-12) for p in each_preds],
                )
                for each_labels, each_preds in zip(y_true, y_pred)
            ]
        )
        return float(res)


class MrrScore(Metric):
    def __init__(self) -> Metric:
        self.name = "mrr"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        mean_mrr = np.mean(
            [
                mrr_score(each_labels, each_preds)
                for each_labels, each_preds in zip(y_true, y_pred)
            ]
        )
        return float(mean_mrr)


class NdcgScore(Metric):
    def __init__(self, k: int):
        self.k = k
        self.name = f"ndcg@{k}"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                ndcg_score(each_labels, each_preds, self.k)
                for each_labels, each_preds in zip(y_true, y_pred)
            ]
        )
        return float(res)


class MetricEvaluator:
    """
    >>> y_true = [[1, 0, 0], [1, 1, 0], [1, 0, 0, 0]]
    >>> y_pred = [[0.2, 0.3, 0.5], [0.18, 0.7, 0.1], [0.18, 0.2, 0.1, 0.1]]

    >>> met_eval = MetricEvaluator(
            labels=y_true,
            predictions=y_pred,
            metric_functions=[
                AucScore(),
                MrrScore(),
                NdcgScore(k=5),
                NdcgScore(k=10),
                LogLossScore(),
                RootMeanSquaredError(),
                AccuracyScore(threshold=0.5),
                F1Score(threshold=0.5),
            ],
        )
    >>> met_eval.evaluate()
    {
        "auc": 0.5555555555555556,
        "mrr": 0.5277777777777778,
        "ndcg@5": 0.7103099178571526,
        "ndcg@10": 0.7103099178571526,
        "logloss": 0.716399020295845,
        "rmse": 0.5022870658128165
        "accuracy": 0.5833333333333334,
        "f1": 0.2222222222222222
    }
    """

    def __init__(
        self,
        labels: list[np.ndarray],
        predictions: list[np.ndarray],
        metric_functions: list[Metric],
    ):
        self.labels = labels
        self.predictions = predictions
        self.metric_functions = metric_functions
        self.evaluations = dict()

    def evaluate(self) -> dict:
        self.evaluations = {
            metric_function.name: metric_function(self.labels, self.predictions)
            for metric_function in self.metric_functions
        }
        return self

    @property
    def metric_functions(self):
        return self.__metric_functions

    @metric_functions.setter
    def metric_functions(self, values):
        invalid_callables = self.__invalid_callables(values)
        if not any(invalid_callables) and invalid_callables:
            self.__metric_functions = values
        else:
            invalid_objects = list(compress(values, invalid_callables))
            invalid_types = [type(item) for item in invalid_objects]
            raise TypeError(f"Following object(s) are not callable: {invalid_types}")

    @staticmethod
    def __invalid_callables(iter: Iterable):
        return [not callable(item) for item in iter]

    def __str__(self):
        if self.evaluations:
            evaluations_json = json.dumps(self.evaluations, indent=4)
            return f"<MetricEvaluator class>: \n {evaluations_json}"
        else:
            return f"<MetricEvaluator class>: {self.evaluations}"

    def __repr__(self):
        return str(self)



from typing import Protocol
import numpy as np


class Metric(Protocol):
    name: str

    def calculate(self, y_true: np.ndarray, y_score: np.ndarray) -> float: ...

    def __str__(self) -> str:
        return f"<Callable Metric: {self.name}>: params: {self.__dict__}"

    def __repr__(self) -> str:
        return str(self)

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self.calculate(y_true, y_score)
