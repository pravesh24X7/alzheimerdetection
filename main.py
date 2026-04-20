import yaml
import os
import joblib

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
from src.utils.logger import custom_logger


def load_config():
    with open("config/config.yaml", mode="r") as f:
        return yaml.safe_load(f)


def main():
    logger = custom_logger()

    logger.info("Loading configuration...")
    config = load_config()

    logger.info("Loading dataset ...")
    df = load_dataset("data/raw/Alzheimer_DataSet.csv")

    logger.info("Dropping null columns")
    df, dropped = drop_null_columns(
        df,
        config["preprocessing"]["null_threshold"]
    )
    logger.info("Dropped columns during preprocessing: %s", dropped)

    if "RID" in df.columns:
        logger.info("Dropping RID column")
        df = df.drop(columns=["RID"])
    
    logger.info("[+] Splitting features and target")
    X, y = split_features_training_testing(df, target=config["data"]["target"])

    logger.info("[+] Splitting dataset into train/test")
    X_train, X_test, y_train, y_test = split_dataset(X, y, config=config)

    logger.info("[+] Encoding target labels")
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train.values)
    y_test = encoder.transform(y_test.values)

    logger.info("[+] Building preprocessing pipeline")
    preprocessor = do_preprocess(X_train)

    logger.info("[+] Loading model")
    model = get_model(config=config)

    logger.info("[+] Building full pipeline")
    pipeline = build_training_pipeline(preprocessor=preprocessor, 
                                       model=model, 
                                       config=config)
    
    logger.info("[+] Training model")
    pipeline.fit(X_train, y_train)

    logger.info("[+] Evaluating model")
    metrics = evalute_model(model=pipeline,
                            X_test=X_test,
                            y_test=y_test)
    
    logger.info("[+] Saving model")
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, f"models/{config['data']['name']}.pkl")
    
    for key in metrics:
        logger.info(f"[*]{key}:{metrics[key]}")


if __name__ == "__main__":
    main()