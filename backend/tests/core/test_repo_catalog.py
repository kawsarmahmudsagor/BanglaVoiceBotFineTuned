"""Every catalog file committed to the repo validates."""
from pathlib import Path

from app.core.catalog import load_catalog

REPO = Path(__file__).resolve().parents[3]


def test_repo_catalog_valid():
    entries, errors = load_catalog(REPO / "catalog")
    assert errors == []
    assert len(entries) >= 1
