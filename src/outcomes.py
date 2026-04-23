import numpy as np
import pandas as pd
from .config import SYNTHETIC

def compute_worker_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    pre = panel[panel["period"] <= 10].groupby("worker_id").agg(
        pre_absorption=("A", "mean"),
        pre_productivity=("P", "mean"),
        pre_ai_use=("U", "mean"),
        pre_monitoring=("M", "mean"),
        pre_redesign=("Z", "mean"),
        skill=("skill", "first"),
        occupation=("occupation", "first")
    ).reset_index()

    shock = panel[panel["shock"] == 1].groupby("worker_id").agg(
        shock_productivity=("P", "mean"),
        shock_quality=("Q", "mean")
    ).reset_index()

    base = pre.merge(shock, on="worker_id", how="left")
    base["productivity_drop"] = base["pre_productivity"] - base["shock_productivity"]

    recovery_records = []
    for wid, g in panel.groupby("worker_id"):
        pre_mean = base.loc[base.worker_id == wid, "pre_productivity"].iloc[0]
        after = g[g["period"] > 13].sort_values("period")
        rec = np.nan
        for _, row in after.iterrows():
            if row["P"] >= pre_mean:
                rec = int(row["period"] - 13)
                break
        if np.isnan(rec):
            rec = int(after["period"].max() - 13 + 1)
        loss = ((pre_mean - g[g["period"] >= 11]["P"]).clip(lower=0)).sum()
        recovery_records.append({"worker_id": wid, "recovery_time": rec, "cumulative_loss": loss})

    recovery = pd.DataFrame(recovery_records)
    out = base.merge(recovery, on="worker_id", how="left")
    return out

def build_outcomes():
    panel = pd.read_csv(SYNTHETIC / "worker_panel.csv")
    outcomes = compute_worker_outcomes(panel)
    outcomes.to_csv(SYNTHETIC / "worker_resilience_outcomes.csv", index=False)
    return outcomes

if __name__ == "__main__":
    build_outcomes()
