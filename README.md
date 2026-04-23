# Agentic AI, Absorption, and Organizational Resilience

Replication package for the synthetic-economy paper:

**Agentic AI, Absorption, and Organizational Resilience**

This repository contains a fully reproducible synthetic-economy pipeline. It generates the worker population, simulates the worker-period panel, computes absorption/productivity/resilience outcomes, estimates the diagnostic regressions, and reproduces the tables and figures used in the manuscript.

The package is designed for journal review and GitHub release. It does **not** use proprietary data. All data are generated synthetically from transparent model assumptions.

## Repository structure

```text
agentic_ai_absorption_replication/
├── README.md
├── LICENSE
├── requirements.txt
├── run_all.py
├── src/
│   ├── config.py
│   ├── simulate.py
│   ├── outcomes.py
│   ├── regressions.py
│   └── figures_tables.py
├── data/
│   ├── parameters/
│   │   ├── occupation_parameters.csv
│   │   └── baseline_parameters.csv
│   └── synthetic/
│       ├── worker_population.csv
│       ├── worker_panel.csv
│       └── worker_resilience_outcomes.csv
├── outputs/
│   ├── figures/
│   └── tables/
└── paper_snippets/
    └── reproducibility_checklist.tex
```

## How to reproduce

From a terminal in the root folder:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

The script will regenerate:

- synthetic worker population;
- worker-period panel;
- worker-level resilience outcomes;
- descriptive tables;
- panel regressions;
- resilience regressions;
- robustness tables;
- paper-ready figures.

## Core model logic

The synthetic economy implements a three-layer agentic architecture:

1. **Agentic AI as bounded productive technology:** AI can accelerate delegable subtasks but may generate quality risk if used without validation.
2. **Workers as endogenous decision-makers:** workers choose AI use, monitoring, and routine redesign each period.
3. **Agentic synthetic economy:** aggregate productivity and resilience emerge from heterogeneous worker decisions under bounded machine quality, fatigue, and complexity shocks.

The central state variable is **AI absorption**, a stock accumulated through validated AI use and routine redesign.

## Main generated variables

- `U`: AI use intensity.
- `M`: monitoring / validation effort.
- `Z`: routine redesign.
- `A`: AI absorption.
- `F`: fatigue.
- `C`: task complexity.
- `L`: workload.
- `E`: error sensitivity.
- `D`: task delegability.
- `Y`: gross useful output.
- `Q`: output quality.
- `Err`: realized error burden.
- `Rework`: rework burden.
- `P`: quality-adjusted productivity.
- `shock`: shock-period indicator.

## Reproducibility note

The synthetic panel is generated with a fixed seed. Changing the seed or calibration parameters will change numerical results but should preserve the qualitative mechanism: AI use becomes economically valuable when transformed into absorption through monitoring and routine redesign.

## Citation note

If this package is uploaded to GitHub or Zenodo, cite the manuscript and repository version used for replication.

## License

MIT License.
