import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    f1_score, recall_score, 
    precision_score
)


def evalute_model(model, X_test: np.ndarray, y_test: np.ndarray):
    y_pred = model.predict(X_test)

    return {
        "accuracy_score": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "recall_score": recall_score(y_test, y_pred, average="weighted")
    }