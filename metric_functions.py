from functools import partial
from typing import Final, Optional, Callable, Iterable

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

from enums import Label

ZERO_DIVISION: Final[int] = 0


def flatten(data: list[list[str]]) -> list[str]:
    return [value for values in data for value in values]


def score_per_label(values: Iterable) -> dict[str, float]:
    return {str(index): value for index, value in enumerate(values)}


def _f1_score(y_true: list[list[str]], y_pred: list[list[str]], average: Optional[str]):
    return f1_score(
        y_true=flatten(y_true),
        y_pred=flatten(y_pred),
        average=average,
        zero_division=ZERO_DIVISION,
        labels=Label.only_punctuations(),
    )


def _precision_score(
    y_true: list[list[str]], y_pred: list[list[str]], average: Optional[str]
):
    return precision_score(
        y_true=flatten(y_true),
        y_pred=flatten(y_pred),
        average=average,
        zero_division=ZERO_DIVISION,
        labels=Label.only_punctuations(),
    )


def _recall_score(
    y_true: list[list[str]], y_pred: list[list[str]], average: Optional[str]
):
    return recall_score(
        y_true=flatten(y_true),
        y_pred=flatten(y_pred),
        average=average,
        zero_division=ZERO_DIVISION,
        labels=Label.only_punctuations(),
    )


def f1_per_label(y_true: list[list[str]], y_pred: list[list[str]]) -> dict[str, float]:
    values = _f1_score(y_true=y_true, y_pred=y_pred, average=None)
    return score_per_label(values=values)


def precision_per_label(
    y_true: list[list[str]], y_pred: list[list[str]]
) -> dict[str, float]:
    values = _precision_score(y_true=y_true, y_pred=y_pred, average=None)
    return score_per_label(values=values)


def recall_per_label(
    y_true: list[list[str]], y_pred: list[list[str]]
) -> dict[str, float]:
    values = _recall_score(y_true=y_true, y_pred=y_pred, average=None)
    return score_per_label(values=values)


def _confusion_matrix(y_true: list[list[str]], y_pred: list[list[str]]):
    return confusion_matrix(
        y_true=flatten(y_true), y_pred=flatten(y_pred), labels=Label.only_punctuations()
    )


METRICS: Final[dict[str, Callable]] = {
    "f1_micro": partial(_f1_score, average="micro"),
    "f1_macro": partial(_f1_score, average="macro"),
    "f1_weighted": partial(_f1_score, average="weighted"),
    "f1_class": f1_per_label,
    "precision_micro": partial(_precision_score, average="micro"),
    "precision_macro": partial(_precision_score, average="macro"),
    "precision_weighted": partial(_precision_score, average="weighted"),
    "precision_class": precision_per_label,
    "recall_micro": partial(_recall_score, average="micro"),
    "recall_macro": partial(_recall_score, average="macro"),
    "recall_weighted": partial(_recall_score, average="weighted"),
    "recall_class": recall_per_label,
    "confusion_matrix": _confusion_matrix,
}
