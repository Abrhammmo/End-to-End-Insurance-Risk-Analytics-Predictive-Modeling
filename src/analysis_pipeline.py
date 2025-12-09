# analysis_pipeline.py

from src.data_loader import load_data
from src.preprocess import basic_clean
from src.hypothesis_tests import chi_square_test, t_test

def run_pipeline(path: str):
    print("Loading data...")
    df = load_data(path)

    print("Cleaning data...")
    df = basic_clean(df)

    results = {}

    # ------------------------------------------
    # Null Hypothesis 1: No risk difference across provinces
    # ------------------------------------------
    print("\nTesting: Risk difference across provinces...")
    results["province_risk"] = chi_square_test("Province", df)

    # ------------------------------------------
    # Null Hypothesis 2: No risk difference across zip codes
    # PostalCode column
    # ------------------------------------------
    print("\nTesting: Risk difference between zip codes...")
    results["zipcode_risk"] = chi_square_test("PostalCode", df)

    # ------------------------------------------
    # Null Hypothesis 3: No margin difference between zip codes
    # ------------------------------------------
    print("\nTesting: Margin difference between zip codes...")
    results["zipcode_margin"] = t_test("PostalCode", df, "Margin")

    # ------------------------------------------
    # Null Hypothesis 4: No risk difference between Men and Women
    # ------------------------------------------
    print("\nTesting: Gender-based risk difference...")
    results["gender_risk"] = chi_square_test("Gender", df)

    return results
