import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def drop_null_columns(df: pd.DataFrame, threshold: float):
    null_ratio = df.isnull().mean()
    cols_to_drop = null_ratio[null_ratio > threshold].index
    return df.drop(columns=cols_to_drop), list(cols_to_drop)


def split_features_training_testing(df: pd.DataFrame, target: str):
    X, y = df.drop(columns=[target]), df[target]
    return X,y


def do_preprocess(X: pd.DataFrame, y=pd.Series):
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = X.select_dtypes(include=["object"]).columns

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns),
    ])

    return preprocess