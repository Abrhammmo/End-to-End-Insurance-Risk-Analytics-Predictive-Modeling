# utils.py

def display_results(results: dict):
    for name, res in results.items():
        print("\n==============================")
        print(f" Test: {name} ")
        print("==============================")
        print(f"Statistic: {res['statistic']:.4f}")
        print(f"P-value : {res['pvalue']:.4f}")

        if res["pvalue"] < 0.05:
            print("➡ Reject the Null Hypothesis (Significant difference)")
        else:
            print("➡ Fail to Reject the Null Hypothesis (No significant difference)")
