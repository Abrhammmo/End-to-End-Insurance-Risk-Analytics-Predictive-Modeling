# data_loader.py

import pandas as pd

def load_data(path: str):
    """
    Load CSV dataset with simple error handling.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Loaded dataset is empty.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at: {path}")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")
