from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_model(config: dict):
    if config["model"]["type"] == "logistic":
        return LogisticRegression(C=config["model"]["C"],
                                  max_iter=config["model"]["max_iter"],
                                  n_jobs=-1)
    elif config["model"]["type"] == "rfc":
        return RandomForestClassifier(n_estimators=config["model"]["n_estimators"], n_jobs=-1)
    else:
        raise ValueError("Unsupported model type")