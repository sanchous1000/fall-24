from pydantic import BaseModel
from tree_id3 import DecisionTreeID3
from typing import Literal
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier


class EvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float


def evaluate_classifier(
    clf: DecisionTreeID3 | DecisionTreeClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_types: list[Literal["categorical", "numeric"]] | None = None,
) -> EvaluationMetrics:
    """
    Evaluate a classifier using various metrics.

    Parameters
    ----------
    clf : DecisionTreeID3 or sklearn classifier
        The classifier to evaluate
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    feature_types : list[str] | None
        List of feature types ("categorical" or "numeric") for DecisionTreeID3.
        Not used for sklearn classifiers.

    Returns
    -------
    EvaluationMetrics
        Object containing accuracy, precision, recall and f1 scores
    """
    # Fit the classifier
    match clf:
        case DecisionTreeID3():
            if feature_types is None:
                raise ValueError("feature_types must be provided for DecisionTreeID3")
            clf.fit(X_train, y_train, feature_types)
            y_pred = clf.predict(X_test)
        case DecisionTreeClassifier():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    return EvaluationMetrics(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1
    )
