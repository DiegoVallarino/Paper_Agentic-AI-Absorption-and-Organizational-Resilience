from src.config import ensure_dirs
from src.simulate import build_all
from src.outcomes import build_outcomes
from src.regressions import build_regressions
from src.figures_tables import build_figures_tables

def main():
    ensure_dirs()
    print("1/4 Generating synthetic worker population and worker-period panel...")
    build_all()
    print("2/4 Computing worker-level resilience outcomes...")
    build_outcomes()
    print("3/4 Estimating diagnostic regressions...")
    build_regressions()
    print("4/4 Producing figures and tables...")
    build_figures_tables()
    print("Done. Outputs are in data/synthetic and outputs/.")

if __name__ == "__main__":
    main()
