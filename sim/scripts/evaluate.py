#!/usr/bin/env python3
"""Run the SimpleICL evaluate command from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simpleicl.cli import evaluate

if __name__ == "__main__":
    evaluate()
