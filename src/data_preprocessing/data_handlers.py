import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def load_data():
    return pd.read_csv(os.path.join(os.environ["DATA_DIR"], "dataset.csv"))

def add_date_features(df):
    pass

if __name__ == "__main__":
    df = load_data()

    a = 3