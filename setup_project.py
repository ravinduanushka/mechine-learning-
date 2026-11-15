"""Create the cardiometabolic-risk project structure.

Usage:  python setup_project.py [target_dir]     (default: current folder)

Safe to re-run: existing files are never overwritten, only missing ones are created.
"""
import sys
from pathlib import Path

# Folders that hold code -> get an __init__.py
PACKAGES = [
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/utils",
    "tests",
]

# Folders that hold data/artifacts -> get a .gitkeep so git tracks the empty folder
KEEP_DIRS = [
    "data/raw",
    "data/interim",
    "data/processed",
    "data/external",
    "models",
    "reports/figures",
    "reports/metrics",
    "reports/tables",
    "configs",
    "notebooks",
]

NOTEBOOKS = [
    "01_data_preprocessing.ipynb",
    "02_eda.ipynb",
    "03_feature_engineering.ipynb",
    "04_feature_selection.ipynb",
    "05_model_training.ipynb",
    "06_evaluation_insights.ipynb",
]
EMPTY_NOTEBOOK = (
    '{"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", '
    '"language": "python", "name": "python3"}, "language_info": {"name": "python"}}, '
    '"nbformat": 4, "nbformat_minor": 5}\n'
)

# Starter modules (name -> one-line docstring)
MODULES = {
    "src/data/load_data.py": "Download / read raw NHANES files.",
    "src/data/preprocess.py": "Cleaning, filtering and cohort building.",
    "src/features/build_features.py": "Feature engineering (activity summaries, intensity, sedentary).",
    "src/features/selectors.py": "Leakage-safe feature selection.",
    "src/models/train.py": "Model training and cross-validation.",
    "src/models/predict.py": "Load a saved model and predict.",
    "src/evaluation/metrics.py": "ROC-AUC, PR-AUC, calibration, thresholds.",
    "src/evaluation/plots.py": "Evaluation and explainability plots.",
}

FILES = {
    "requirements.txt": (
        "numpy>=1.24.0\npandas>=2.0.0\nscipy>=1.10.0\nscikit-learn>=1.3.0\n"
        "pyarrow>=14.0.0\npyreadstat>=1.2.0\nmatplotlib>=3.7.0\nseaborn>=0.12.0\n"
        "joblib>=1.3.0\npyyaml>=6.0\njupyter>=1.0.0\ntqdm>=4.65.0\nrequests>=2.31.0\n"
        "pytest>=7.4.0\n"
    ),
    ".gitignore": (
        "# data (keep folders, not contents)\ndata/raw/*\ndata/interim/*\ndata/external/*\n"
        "!data/**/.gitkeep\n*.csv\n*.xpt\n\n# artifacts\nmodels/*.joblib\nmodels/*.pkl\n\n"
        "# python / jupyter\n__pycache__/\n.ipynb_checkpoints/\n.pytest_cache/\n.env\nvenv/\n.venv/\n"
    ),
    "configs/config.yaml": (
        "project:\n  random_state: 42\n  test_size: 0.2\n  cv_folds: 5\n"
        "  target: metabolic_syndrome_target\n\n"
        "paths:\n  raw: data/raw\n  interim: data/interim\n  processed: data/processed\n"
        "  models: models\n  figures: reports/figures\n  metrics: reports/metrics\n\n"
        "features:\n  correlation_threshold: 0.9\n  mi_top_k: 30\n  l1_C: 0.1\n"
        "  kmeans_k: 4\n  pca_components: 3\n"
    ),
    "README.md": (
        "# Cardiometabolic Risk Prediction\n\n"
        "Pipeline: `data/raw` -> `src/data` -> `data/processed` -> `src/features` -> "
        "`src/models` -> `models/` + `reports/`.\n\n"
        "```\npython -m pip install -r requirements.txt\npython -m pytest tests -q\n```\n"
    ),
    "src/utils/paths.py": (
        '"""Project paths shared by all modules."""\nfrom pathlib import Path\n\n'
        "ROOT = Path(__file__).resolve().parents[2]\nDATA = ROOT / \"data\"\n"
        "RAW, INTERIM, PROCESSED = DATA / \"raw\", DATA / \"interim\", DATA / \"processed\"\n"
        "MODELS = ROOT / \"models\"\nREPORTS = ROOT / \"reports\"\nCONFIG = ROOT / \"configs\" / \"config.yaml\"\n"
    ),
    "tests/test_smoke.py": (
        '"""Smoke test: the package layout imports."""\n\n\n'
        "def test_imports():\n    import src.data, src.features, src.models, src.evaluation, src.utils\n"
    ),
}


def write_if_missing(path: Path, text: str, created: list, skipped: list):
  if path.exists():
    skipped.append(path)
  else:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    created.append(path)


def main(root: Path):
  created, skipped = [], []
  for d in PACKAGES:
    write_if_missing(root / d / "__init__.py", "", created, skipped)
  for d in KEEP_DIRS:
    write_if_missing(root / d / ".gitkeep", "", created, skipped)
  for nb in NOTEBOOKS:
    write_if_missing(root / "notebooks" / nb, EMPTY_NOTEBOOK, created, skipped)
  for rel, doc in MODULES.items():
    write_if_missing(root / rel, f'"""{doc}"""\n', created, skipped)
  for rel, text in FILES.items():
    write_if_missing(root / rel, text, created, skipped)

  print(f"Project root: {root.resolve()}")
  print(f"created {len(created)} files, left {len(skipped)} existing files untouched")
  for p in created:
    print("  +", p.relative_to(root))


if __name__ == "__main__":
  main(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
