import pandas as pd


def load_dataset(filepath: str="") -> pd.DataFrame:
    return pd.read_csv(filepath)
