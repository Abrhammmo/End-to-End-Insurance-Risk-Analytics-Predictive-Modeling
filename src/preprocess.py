# preprocess.py

import pandas as pd

def basic_clean(df: pd.DataFrame):
    """
    Perform basic preprocessing:
    - Remove duplicates
    - Drop rows where TotalPremium or TotalClaims is missing
    """
    df = df.drop_duplicates()

    required = ["TotalPremium", "TotalClaims"]
    df = df.dropna(subset=required)

    # Create Claim Frequency (1 if claim > 0 else 0)
    df["HasClaim"] = df["TotalClaims"].apply(lambda x: 1 if x > 0 else 0)

    # Claim Severity: meaningful only if claim occurred
    df["ClaimSeverity"] = df.apply(
        lambda row: row["TotalClaims"] if row["HasClaim"] == 1 else 0, axis=1
    )

    # Margin
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

    return df
