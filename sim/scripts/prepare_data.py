#!/usr/bin/env python3
"""Run the SimpleICL prepare data command from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simpleicl.cli import prepare_data

if __name__ == "__main__":
    prepare_data()
