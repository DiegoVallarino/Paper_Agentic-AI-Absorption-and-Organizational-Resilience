import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from .config import SYNTHETIC, TABLES

def stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

def fit_cluster(formula, data):
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["worker_id"]}
    )
    return model

def fit_ols(formula, data):
    return smf.ols(formula, data=data).fit()

def compact_regression_table(models, variables):
    rows = []
    for var in variables:
        coef_row = {"Variable": var}
        se_row = {"Variable": ""}
        for idx, model in enumerate(models, start=1):
            name = f"({idx})"
            if var in model.params.index:
                coef = model.params[var]
                pval = model.pvalues[var]
                se = model.bse[var]
                coef_row[name] = f"{coef:.3f}{stars(pval)}"
                se_row[name] = f"({se:.3f})"
            else:
                coef_row[name] = ""
                se_row[name] = ""
        rows.append(coef_row)
        rows.append(se_row)
    r2 = {"Variable": "R-squared"}
    obs = {"Variable": "Observations"}
    for idx, model in enumerate(models, start=1):
        r2[f"({idx})"] = f"{model.rsquared:.3f}"
        obs[f"({idx})"] = f"{int(model.nobs):,}"
    rows.append(r2)
    rows.append(obs)
    return pd.DataFrame(rows)

def estimate_panel_models():
    panel = pd.read_csv(SYNTHETIC / "worker_panel.csv")
    panel["UxM"] = panel["U"] * panel["M"]
    panel["UxZ"] = panel["U"] * panel["Z"]
    panel["AxShock"] = panel["A"] * panel["shock"]
    # Avoid Patsy conflict between the variable named C and the categorical function C().
    panel = panel.rename(columns={"C": "complexity", "L": "workload", "F": "fatigue"})

    f1 = "P ~ A + shock + U + M + Z + complexity + workload + fatigue + C(occupation) + C(period)"
    f2 = "P ~ A + shock + AxShock + U + M + Z + complexity + workload + fatigue + C(occupation) + C(period)"
    f3 = "P ~ A + shock + U + M + Z + UxM + UxZ + complexity + workload + fatigue + C(occupation) + C(period)"
    f4 = "P ~ A + shock + AxShock + U + M + Z + UxM + UxZ + complexity + workload + fatigue + C(occupation) + C(period)"
    models = [fit_cluster(f, panel) for f in [f1, f2, f3, f4]]

    variables = ["A", "shock", "AxShock", "U", "M", "Z", "UxM", "UxZ", "complexity", "workload", "fatigue"]
    table = compact_regression_table(models, variables)
    table = table.replace({
        "A": "AI absorption",
        "shock": "Shock indicator",
        "AxShock": "Absorption x shock",
        "U": "AI use intensity",
        "M": "Monitoring effort",
        "Z": "Routine redesign",
        "UxM": "AI use x monitoring",
        "UxZ": "AI use x redesign",
        "complexity": "Complexity",
        "workload": "Workload",
        "fatigue": "Fatigue"
    })
    table.to_csv(TABLES / "main_panel_regressions.csv", index=False)
    return models, table

def estimate_resilience_models():
    outcomes = pd.read_csv(SYNTHETIC / "worker_resilience_outcomes.csv")
    f1 = "productivity_drop ~ pre_absorption + pre_productivity + skill + C(occupation)"
    f2 = "recovery_time ~ pre_absorption + pre_productivity + skill + C(occupation)"
    f3 = "cumulative_loss ~ pre_absorption + pre_productivity + skill + C(occupation)"
    models = [fit_ols(f, outcomes) for f in [f1, f2, f3]]
    variables = ["pre_absorption", "pre_productivity", "skill"]
    table = compact_regression_table(models, variables)
    table = table.replace({
        "pre_absorption": "Pre-shock absorption",
        "pre_productivity": "Pre-shock productivity",
        "skill": "Baseline skill"
    })
    table.to_csv(TABLES / "resilience_regressions.csv", index=False)
    return models, table

def build_regressions():
    TABLES.mkdir(parents=True, exist_ok=True)
    estimate_panel_models()
    estimate_resilience_models()

if __name__ == "__main__":
    build_regressions()
