from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_model(config: dict):
    if config["type"] == "logistic":
        return LogisticRegression(C=config["C"],
                                  max_iter=config["max_iter"],
                                  n_jobs=-1)
    elif config["type"] == "rfc":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1)
    else:
        raise ValueError("Unsupported model type")