# ============================================================
# data_processing.py  (FINAL VERSION FOR YOUR DATASET)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# ============================================================
# COLUMN DEFINITIONS (YOUR EXACT SCHEMA)
# ============================================================

NUMERIC_COLS = [
    'UnderwrittenCoverID', 'PolicyID', 'PostalCode', 'mmcode',
    'RegistrationYear', 'Cylinders', 'cubiccapacity', 'kilowatts',
    'NumberOfDoors', 'CustomValueEstimate', 'NumberOfVehiclesInFleet',
    'SumInsured', 'CalculatedPremiumPerTerm', 'TotalPremium', 'TotalClaims'
]

CATEGORICAL_COLS = [
    'Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 'AccountType',
    'MaritalStatus', 'Gender', 'Country', 'Province', 'MainCrestaZone',
    'SubCrestaZone', 'ItemType', 'VehicleType', 'make', 'Model', 'bodytype',
    'VehicleIntroDate', 'AlarmImmobiliser', 'TrackingDevice',
    'CapitalOutstanding', 'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted',
    'CrossBorder', 'TermFrequency', 'ExcessSelected', 'CoverCategory',
    'CoverType', 'CoverGroup', 'Section', 'Product', 'StatutoryClass',
    'StatutoryRiskType'
]

BOOLEAN_COLS = ['IsVATRegistered']

DATE_COLS = ['TransactionMonth', 'VehicleIntroDate']


# ============================================================
# 1. LOAD + CLEAN DATA
# ============================================================

def load_and_clean_data(filepath):

    df = pd.read_csv(filepath)

    # Remove duplicate rows
    df = df.drop_duplicates(keep="first")

    # Clean whitespace for all string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Convert date columns
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert boolean columns to 0/1
    for col in BOOLEAN_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    return df


# ============================================================
# 2. ENCODER FUNCTION
# ============================================================

def encoder(method, df):

    df_encoded = df.copy()

    # Ensure only available columns are encoded
    label_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    # ONE-HOT recommended for province/vehicle regions
    onehot_target_cols = ['Province', 'PostalCode', 'Bank', 'VehicleType', 'make', 'Model']
    onehot_target_cols = [c for c in onehot_target_cols if c in df.columns]

    if method == "labelEncoder":
        for col in label_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        return df_encoded

    elif method == "oneHotEncoder":
        df_encoded = pd.get_dummies(
            df_encoded,
            columns=onehot_target_cols,
            drop_first=True,
            dtype='int8'
        )
        return df_encoded

    return df_encoded


# ============================================================
# 3. SCALER FUNCTION
# ============================================================

def scaler(method, df):

    df_scaled = df.copy()

    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]

    if method == "standardScaler":
        scaler = StandardScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        return df_scaled

    elif method == "minMaxScaler":
        scaler = MinMaxScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        return df_scaled

    elif method == "npLog":
        df_scaled[numeric_cols] = np.log(df_scaled[numeric_cols] + 1)
        return df_scaled

    return df_scaled
