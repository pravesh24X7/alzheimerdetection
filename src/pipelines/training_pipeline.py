from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


def build_training_pipeline(preprocessor, model, config):
    steps = [("preprocessing", preprocessor)]

    if config["feature_selection"]["enabled"]:
        selector = SelectFromModel(
            LogisticRegression(penalty="l1", solver="liblinear", C=0.1)
        )

        steps.append(("feature_selection", selector))
    
    if config["pca"]["enabled"]:
        steps.append(("pca", PCA(n_components=config["pca"]["variance"])))
    
    steps.append(("model", model))
    return Pipeline(steps)


def split_dataset(X, y, config):
    return train_test_split(X, y,
                            test_size=config["data"]["test_size"],
                            stratify=y,
                            random_state=config["data"]["random_state"])