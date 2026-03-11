#!/usr/bin/env python
"""Unified command-line interface for shortcut-learning pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Shortcut Learning Pipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ["preprocess", "train", "evaluate", "test_shortcuts", "visualize"]:
        sp = sub.add_parser(name)
        sp.add_argument("args", nargs=argparse.REMAINDER)

    ns = parser.parse_args()
    script = str(Path(__file__).parent / f"{ns.cmd}.py")
    cmd = [sys.executable, script] + ns.args
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
