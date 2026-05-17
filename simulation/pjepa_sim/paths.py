"""Repository-local paths used by simulation commands."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SIMULATION_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PACKAGE_ROOT / "benchmark" / "configs"
OUTPUT_DIR = SIMULATION_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
