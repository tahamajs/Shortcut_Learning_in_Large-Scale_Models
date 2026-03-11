#!/usr/bin/env python
"""Unified command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ["preprocess", "train", "evaluate", "test_shortcuts"]:
        sp = sub.add_parser(name)
        sp.add_argument("args", nargs=argparse.REMAINDER)

    ns = parser.parse_args()
    script = f"scripts/{ns.cmd}.py"
    cmd = [sys.executable, script] + ns.args
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
