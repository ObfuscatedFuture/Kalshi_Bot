from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"

# Collect Fed presser PDFs dynamically.
fed_paths = [str(p) for p in sorted(DATASET_DIR.glob("*.pdf"))]
