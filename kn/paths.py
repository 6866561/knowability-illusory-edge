"""Data locations that work both in the public repository (data/ inside the repo) and in the original layout (data/ beside it)."""
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data" if (REPO / "data" / "d7_audit_results.csv").exists() else REPO.parent / "data"
RAW = REPO.parent / "raw"
