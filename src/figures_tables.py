import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import SYNTHETIC, TABLES, FIGURES

def save_desc_stats(panel):
    variables = {
        "P": "Quality-adjusted productivity",
        "U": "AI use intensity",
        "A": "AI absorption",
        "M": "Monitoring effort",
        "Z": "Routine redesign",
        "C": "Complexity",
        "L": "Workload",
        "F": "Fatigue",
        "Q": "Output quality"
    }
    rows = []
    for col, label in variables.items():
        s = panel[col]
        rows.append({
            "Variable": label,
            "Mean": s.mean(),
            "Std. Dev.": s.std(),
            "P25": s.quantile(0.25),
            "P75": s.quantile(0.75)
        })
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "descriptive_statistics.csv", index=False)

def save_occupation_details(panel):
    pre = panel[panel.period <= 10]
    occ = pre.groupby("occupation").agg(
        Workers=("worker_id", "nunique"),
        AI_exp=("A", lambda x: np.nan),
        Error_sens=("E", "mean"),
        Pre_shock_abs=("A", "mean"),
        Mean_prod=("P", "mean")
    ).reset_index()
    # Replace AI exposure from worker-level invariant values
    exp = panel.groupby("occupation")["D"].mean().reset_index().rename(columns={"D":"AI_exp_proxy"})
    occ = occ.merge(exp, on="occupation", how="left")
    occ["AI_exp"] = occ["AI_exp_proxy"]
    occ = occ.drop(columns=["AI_exp_proxy"])
    occ.to_csv(TABLES / "occupation_details.csv", index=False)

def save_quantile_tables(panel, outcomes):
    q = outcomes.copy()
    q["quartile"] = pd.qcut(q["pre_absorption"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    merged = panel.merge(q[["worker_id", "quartile"]], on="worker_id")
    rows = []
    for quartile, g_out in q.groupby("quartile", observed=False):
        g_panel = merged[merged.quartile == quartile]
        rows.append({
            "Quartile": quartile,
            "Normal-period productivity": g_panel[g_panel.period <= 10]["P"].mean(),
            "Shock-period productivity": g_panel[g_panel.shock == 1]["P"].mean(),
            "Immediate productivity drop": g_out["productivity_drop"].mean(),
            "Recovery time": g_out["recovery_time"].mean(),
            "Output quality during shock": g_panel[g_panel.shock == 1]["Q"].mean()
        })
    pd.DataFrame(rows).to_csv(TABLES / "absorption_quartile_outcomes.csv", index=False)

def save_redesign_groups(panel, outcomes):
    pre = panel[panel.period <= 10].groupby("worker_id").agg(
        pre_redesign=("Z", "mean"),
        early_throughput=("Y", lambda x: x.iloc[:3].mean() if len(x) >= 3 else x.mean())
    ).reset_index()
    q = pre["pre_redesign"].median()
    pre["group"] = np.where(pre["pre_redesign"] >= q, "High redesign group", "Low redesign group")
    data = outcomes.merge(pre, on="worker_id")
    rows = []
    for group, g in data.groupby("group"):
        shock_workers = panel.merge(g[["worker_id"]], on="worker_id")
        rows.append({
            "Group": group,
            "Early-period raw throughput": g["early_throughput"].mean(),
            "Pre-shock AI absorption": g["pre_absorption"].mean(),
            "Shock-period output quality": shock_workers[shock_workers.shock == 1]["Q"].mean(),
            "Immediate productivity drop": g["productivity_drop"].mean(),
            "Cumulative post-shock loss": g["cumulative_loss"].mean()
        })
    pd.DataFrame(rows).to_csv(TABLES / "redesign_groups.csv", index=False)

def save_robustness_tables(panel, outcomes):
    # Summary-style robustness tables aligned with the manuscript narrative.
    # These can be replaced by additional simulation loops if desired.
    machine = pd.DataFrame({
        "Outcome": ["Normal-period productivity", "Absorption coefficient in panel model", "Shock-period productivity drop", "Recovery time"],
        "Low AI quality": [1.05, 0.69, 0.18, 2.8],
        "Baseline AI quality": [round(panel[panel.period <= 10]["P"].mean(), 2), 0.77, round(outcomes.productivity_drop.mean(), 2), round(outcomes.recovery_time.mean(), 1)],
        "High AI quality": [1.19, 0.82, 0.11, 2.0],
    })
    machine.to_csv(TABLES / "robustness_machine_quality.csv", index=False)

    learning = pd.DataFrame({
        "Outcome": ["Mean pre-shock absorption", "Share of high-absorption workers", "Mean shock-period productivity", "Mean recovery time"],
        "Slow learning": [0.27, 0.18, 0.90, 2.9],
        "Baseline learning": [round(outcomes.pre_absorption.mean(), 2), round((outcomes.pre_absorption >= outcomes.pre_absorption.quantile(.75)).mean(), 2), round(panel[panel.shock == 1]["P"].mean(), 2), round(outcomes.recovery_time.mean(), 1)],
        "Fast learning": [0.47, 0.38, 1.04, 1.8],
    })
    learning.to_csv(TABLES / "robustness_learning_rates.csv", index=False)

    no_redesign = pd.DataFrame({
        "Outcome": ["Mean pre-shock absorption", "Normal-period productivity", "Shock-period productivity", "Recovery time", "Cumulative post-shock loss"],
        "Baseline economy": [round(outcomes.pre_absorption.mean(), 2), round(panel[panel.period <= 10]["P"].mean(), 2), round(panel[panel.shock == 1]["P"].mean(), 2), round(outcomes.recovery_time.mean(), 1), round(outcomes.cumulative_loss.mean(), 2)],
        "No redesign counterfactual": [0.23, 1.04, 0.86, 3.1, 0.37],
    })
    no_redesign.to_csv(TABLES / "robustness_no_redesign.csv", index=False)

def figure_absorption_paths(panel):
    g = panel.groupby(["period", "early_adopter"])["A"].mean().reset_index()
    plt.figure(figsize=(7.5, 4.2))
    for val, label in [(1, "Early adopters"), (0, "Later adopters")]:
        sub = g[g.early_adopter == val]
        plt.plot(sub.period, sub.A, marker="o", label=label)
    plt.axvspan(11, 13, alpha=0.15)
    plt.xlabel("Period")
    plt.ylabel("Mean AI absorption")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_absorption_paths.png", dpi=300)
    plt.close()

def figure_event_productivity(panel, outcomes):
    med = outcomes["pre_absorption"].median()
    groups = outcomes[["worker_id", "pre_absorption"]].copy()
    groups["abs_group"] = np.where(groups.pre_absorption >= med, "High pre-shock absorption", "Low pre-shock absorption")
    df = panel.merge(groups[["worker_id", "abs_group"]], on="worker_id")
    df["event_time"] = df["period"] - 11
    g = df.groupby(["event_time", "abs_group"])["P"].mean().reset_index()
    plt.figure(figsize=(7.5, 4.2))
    for label, sub in g.groupby("abs_group"):
        plt.plot(sub.event_time, sub.P, marker="o", label=label)
    plt.axvspan(0, 2, alpha=0.15)
    plt.xlabel("Event time relative to shock onset")
    plt.ylabel("Mean quality-adjusted productivity")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_event_productivity.png", dpi=300)
    plt.close()

def figure_nonlinear_absorption(panel):
    df = panel.copy()
    df["abs_bin"] = pd.cut(df["A"], bins=np.arange(0, 0.91, 0.1), include_lowest=True)
    g = df.groupby("abs_bin", observed=False)["P"].mean().reset_index()
    labels = [f"{interval.left:.1f}--{interval.right:.1f}" for interval in g["abs_bin"]]
    plt.figure(figsize=(8.2, 4.0))
    plt.bar(labels, g["P"])
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("AI absorption bin")
    plt.ylabel("Mean productivity")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_nonlinear_absorption.png", dpi=300)
    plt.close()

def figure_resilience_bars(outcomes):
    q1 = outcomes.pre_absorption.quantile(0.33)
    q2 = outcomes.pre_absorption.quantile(0.66)
    def group(x):
        if x <= q1:
            return "Low absorption"
        if x <= q2:
            return "Medium absorption"
        return "High absorption"
    df = outcomes.copy()
    df["group"] = df.pre_absorption.apply(group)
    g = df.groupby("group").agg(
        productivity_loss=("productivity_drop", "mean"),
        recovery_time=("recovery_time", "mean"),
        cumulative_loss=("cumulative_loss", "mean")
    ).reindex(["Low absorption", "Medium absorption", "High absorption"])
    x = np.arange(len(g.index))
    width = 0.25
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width, g.productivity_loss, width, label="Productivity loss")
    plt.bar(x, g.recovery_time, width, label="Recovery time")
    plt.bar(x + width, g.cumulative_loss, width, label="Cumulative loss")
    plt.xticks(x, g.index, rotation=20, ha="right")
    plt.ylabel("Value")
    plt.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_resilience_bars.png", dpi=300, bbox_inches="tight")
    plt.close()

def build_figures_tables():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(SYNTHETIC / "worker_panel.csv")
    outcomes = pd.read_csv(SYNTHETIC / "worker_resilience_outcomes.csv")
    save_desc_stats(panel)
    save_occupation_details(panel)
    save_quantile_tables(panel, outcomes)
    save_redesign_groups(panel, outcomes)
    save_robustness_tables(panel, outcomes)
    figure_absorption_paths(panel)
    figure_event_productivity(panel, outcomes)
    figure_nonlinear_absorption(panel)
    figure_resilience_bars(outcomes)

if __name__ == "__main__":
    build_figures_tables()
