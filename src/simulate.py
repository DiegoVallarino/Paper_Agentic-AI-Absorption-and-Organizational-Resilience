import numpy as np
import pandas as pd
from pathlib import Path
from .config import SEED, N_WORKERS, T_PERIODS, SHOCK_PERIODS, PARAMS, SYNTHETIC

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def clamp(x, lo=0.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)

def occupation_parameters():
    rows = [
        ("Analyst", 280, 0.74, 0.72, 1.08, 1.04, 0.73, 0.58),
        ("Developer", 245, 0.81, 0.66, 1.05, 1.03, 0.80, 0.62),
        ("Support specialist", 265, 0.59, 0.54, 0.88, 1.22, 0.64, 0.52),
        ("Manager", 215, 0.51, 0.63, 0.96, 1.02, 0.52, 0.55),
        ("Researcher", 245, 0.79, 0.70, 1.14, 0.99, 0.76, 0.61),
        ("Compliance/legal review", 250, 0.69, 0.82, 1.10, 1.00, 0.68, 0.57),
    ]
    return pd.DataFrame(rows, columns=[
        "occupation", "workers", "ai_exposure", "error_sensitivity",
        "complexity_mean", "workload_mean", "delegability", "skill_mean"
    ])

def baseline_parameters():
    rows = [
        ("alpha_1_absorbed_ai_output", 1.15, "Output gain from absorbed AI use"),
        ("alpha_2_redesign_output", 0.25, "Output gain from routine redesign"),
        ("alpha_3_complexity_penalty", 0.55, "Throughput penalty from complexity"),
        ("q_1_skill_quality", 0.18, "Quality gain from baseline skill"),
        ("q_2_monitoring_quality", 0.18, "Quality gain from monitoring"),
        ("q_3_absorption_quality", 0.12, "Quality gain from absorption"),
        ("q_4_unmonitored_ai_penalty", 0.22, "Quality penalty from unvalidated AI use"),
        ("q_5_complexity_error_penalty", 0.10, "Quality penalty from complexity and error sensitivity"),
        ("rho_1_validated_use_learning", 0.075, "Absorption gain from validated AI use"),
        ("rho_2_redesign_learning", 0.045, "Absorption gain from redesign"),
        ("rho_3_skill_learning", 0.006, "Absorption gain from skill"),
        ("rho_4_error_learning_penalty", 0.030, "Absorption loss from errors"),
        ("phi_1_fatigue_persistence", 0.52, "Fatigue persistence"),
        ("phi_2_task_burden_fatigue", 0.085, "Fatigue accumulation from complexity and workload"),
        ("phi_3_absorbed_ai_relief", 0.050, "Fatigue relief from absorbed AI use"),
        ("shock_complexity_delta", 0.32, "Complexity shock increment"),
        ("shock_workload_delta", 0.25, "Workload shock increment"),
        ("shock_deadline_delta", 0.20, "Deadline pressure shock increment"),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "interpretation"])

def build_worker_population(seed=SEED):
    rng = np.random.default_rng(seed)
    occ = occupation_parameters()
    workers = []
    worker_id = 1
    for _, row in occ.iterrows():
        for _ in range(int(row["workers"])):
            skill = clamp(rng.normal(row["skill_mean"], 0.12), 0.15, 0.95)
            digital = clamp(rng.normal(0.55 + 0.25*(row["ai_exposure"]-0.6), 0.16), 0.05, 0.95)
            trust = clamp(rng.normal(0.52 + 0.20*(row["ai_exposure"]-0.6), 0.16), 0.05, 0.95)
            verification = clamp(rng.normal(0.54 + 0.18*(row["error_sensitivity"]-0.6), 0.14), 0.05, 0.95)
            adaptability = clamp(rng.normal(0.48 + 0.18*digital, 0.15), 0.05, 0.95)
            early_adopter = int(rng.uniform() < sigmoid(-0.4 + 1.1*digital + 0.7*trust + 0.5*row["ai_exposure"]))
            workers.append({
                "worker_id": worker_id,
                "occupation": row["occupation"],
                "ai_exposure": row["ai_exposure"],
                "baseline_error_sensitivity": row["error_sensitivity"],
                "baseline_complexity": row["complexity_mean"],
                "baseline_workload": row["workload_mean"],
                "baseline_delegability": row["delegability"],
                "skill": skill,
                "digital_capability": digital,
                "trust_ai": trust,
                "verification_discipline": verification,
                "adaptability": adaptability,
                "early_adopter": early_adopter
            })
            worker_id += 1
    return pd.DataFrame(workers)

def simulate_panel(population, seed=SEED):
    rng = np.random.default_rng(seed + 1000)
    records = []
    states = {}
    for _, w in population.iterrows():
        a0 = clamp(0.05 + 0.08*w.digital_capability + 0.04*w.skill + rng.normal(0, 0.02), 0.02, 0.25)
        f0 = clamp(0.05 + rng.normal(0, 0.02), 0.0, 0.20)
        states[int(w.worker_id)] = {"A": float(a0), "F": float(f0), "last_P": np.nan}

    for t in range(1, T_PERIODS + 1):
        shock = int(t in SHOCK_PERIODS)
        for _, w in population.iterrows():
            wid = int(w.worker_id)
            A = states[wid]["A"]
            F = states[wid]["F"]

            C = clamp(rng.normal(w.baseline_complexity + 0.32*shock, 0.13), 0.25, 1.90)
            L = clamp(rng.normal(w.baseline_workload + 0.25*shock, 0.12), 0.35, 1.90)
            E = clamp(rng.normal(w.baseline_error_sensitivity, 0.08), 0.20, 0.98)
            D = clamp(rng.normal(w.baseline_delegability, 0.08), 0.20, 0.98)
            deadline = 0.20 * shock

            adoption_shift = 0.30 if (w.early_adopter or t >= 6) else -0.35
            learning_trend = min(t, 10) * 0.025
            U = sigmoid(-0.65 + adoption_shift + 1.00*w.trust_ai + 0.75*w.digital_capability
                        + 0.65*A + 0.55*D - 0.22*C - 0.35*F + learning_trend
                        + rng.normal(0, 0.30))
            U = clamp(U)

            M = sigmoid(-0.90 + 1.25*w.verification_discipline + 0.70*w.skill
                        + 0.50*E + 0.20*C + 0.45*U + 0.30*A - 0.25*deadline
                        + rng.normal(0, 0.30))
            M = clamp(M)

            Z = sigmoid(-1.20 + 1.25*w.adaptability + 0.60*w.digital_capability
                        + 0.35*A + 0.35*U*M - 0.20*shock + rng.normal(0, 0.32))
            Z = clamp(Z)

            # Production
            alpha1, alpha2, alpha3 = 1.15, 0.25, 0.55
            Y = ((w.skill + alpha1*U*A*D + alpha2*Z) * (1-F) * L) / (1 + alpha3*C)

            Q = (0.72 + 0.18*w.skill + 0.18*M + 0.12*A
                 - 0.22*U*(1-M) - 0.10*C*E + rng.normal(0, 0.025))
            Q = clamp(Q, 0.35, 1.05)

            Err = max(0, 0.28*U*(1-M) + 0.12*C*E - 0.18*A + rng.normal(0, 0.025))
            Rework = max(0, 0.30*Err + 0.30*E*Err)

            lambda_R = 0.60
            lambda_M = 0.18 * (1 + 0.50*deadline)
            lambda_Z = 0.20 * (1 + 0.60*deadline)
            P = (Y*Q) / (1 + lambda_R*Rework + lambda_M*M + lambda_Z*Z)

            # Normalize productivity around paper-friendly scale
            P_scaled = 1.00 + 0.55*(P - 0.55)

            A_next = clamp(A + 0.075*U*M + 0.045*Z + 0.006*w.skill - 0.030*Err + rng.normal(0, 0.008))
            F_next = clamp(0.52*F + 0.085*C*L - 0.050*A*U + rng.normal(0, 0.010))

            records.append({
                "worker_id": wid,
                "period": t,
                "occupation": w.occupation,
                "shock": shock,
                "early_adopter": int(w.early_adopter),
                "skill": w.skill,
                "digital_capability": w.digital_capability,
                "trust_ai": w.trust_ai,
                "verification_discipline": w.verification_discipline,
                "adaptability": w.adaptability,
                "C": C, "L": L, "E": E, "D": D,
                "U": U, "M": M, "Z": Z,
                "A": A, "F": F,
                "Y": Y, "Q": Q, "Err": Err, "Rework": Rework,
                "P": P_scaled
            })

            states[wid]["A"] = float(A_next)
            states[wid]["F"] = float(F_next)
            states[wid]["last_P"] = float(P_scaled)

    return pd.DataFrame(records)

def build_all():
    PARAMS.mkdir(parents=True, exist_ok=True)
    SYNTHETIC.mkdir(parents=True, exist_ok=True)
    occupation_parameters().to_csv(PARAMS / "occupation_parameters.csv", index=False)
    baseline_parameters().to_csv(PARAMS / "baseline_parameters.csv", index=False)
    population = build_worker_population()
    panel = simulate_panel(population)
    population.to_csv(SYNTHETIC / "worker_population.csv", index=False)
    panel.to_csv(SYNTHETIC / "worker_panel.csv", index=False)
    return population, panel

if __name__ == "__main__":
    build_all()
