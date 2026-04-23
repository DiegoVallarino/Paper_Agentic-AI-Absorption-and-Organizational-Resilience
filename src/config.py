from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PARAMS = DATA / "parameters"
SYNTHETIC = DATA / "synthetic"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

SEED = 20260418
N_WORKERS = 1500
T_PERIODS = 20
SHOCK_PERIODS = {11, 12, 13}

def ensure_dirs():
    for path in [PARAMS, SYNTHETIC, FIGURES, TABLES]:
        path.mkdir(parents=True, exist_ok=True)
