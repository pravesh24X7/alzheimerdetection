import yaml

from sklearn.preprocessing import LabelEncoder
from src.data.load_data import load_dataset
from src.features.preprocess import (
    drop_null_columns,
    do_preprocess,
    split_features_training_testing
)
from src.models.train_model import get_model
from src.models.evaluate import evalute_model
from src.pipelines.training_pipeline import (
    build_training_pipeline,
    split_dataset
)


def load_config():
    with open("config/config.yaml", mode="r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_dataset("data/raw/Alzheimer_DataSet.csv")
    df, dropped = drop_null_columns(
        df,
        config["preprocessing"]["null_threshold"]
    )

    print("[*] Following columns are dopped while preprocessing:\n", dropped)

    if "RID" in df.columns:
        df = df.drop(columns=["RID"])
    
    X, y = split_features_training_testing(df, target=config["data"]["target"])
    X_train, X_test, y_train, y_test = split_dataset(X, y, config=config)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train.values)
    y_test = encoder.transform(y_test.values)

    preprocessor = do_preprocess(X_train)
    model = get_model(config=config)

    pipeline = build_training_pipeline(preprocessor=preprocessor, 
                                       model=model, 
                                       config=config)
    pipeline.fit(X_train, y_train)

    metrics = evalute_model(model=pipeline,
                            X_test=X_test,
                            y_test=y_test)
    print(metrics)


if __name__ == "__main__":
    main()