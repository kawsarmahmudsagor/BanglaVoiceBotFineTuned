"""Every catalog file committed to the repo validates."""
from pathlib import Path

from mlops_shared.catalog import load_catalog

REPO = Path(__file__).resolve().parents[2]


def test_repo_catalog_valid():
    entries, errors = load_catalog(REPO / "catalog")
    assert errors == []
    assert len(entries) >= 1
