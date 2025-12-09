# hypothesis_tests.py

import pandas as pd
import numpy as np
from scipy.stats import chisquare, chi2_contingency, ttest_ind

def chi_square_test(group_col, df):
    """
    Chi-square test for categorical KPIs (Claim Frequency)
    """
    contingency = pd.crosstab(df[group_col], df["HasClaim"])

    if contingency.shape[0] < 2:
        raise ValueError(f"Not enough groups in {group_col}")

    stat, p, dof, expected = chi2_contingency(contingency)
    return {"statistic": stat, "pvalue": p, "table": contingency}


def t_test(group_col, df, kpi):
    """
    t-test for numerical metrics:
    - Claim Severity
    - Margin
    """
    groups = df[group_col].unique()
    if len(groups) < 2:
        raise ValueError(f"Not enough groups in {group_col} to run t-test")

    g1 = df[df[group_col] == groups[0]][kpi]
    g2 = df[df[group_col] == groups[1]][kpi]

    stat, p = ttest_ind(g1, g2, equal_var=False)

    return {"statistic": stat, "pvalue": p, "groups": groups}
