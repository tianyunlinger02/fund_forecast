#!/usr/bin/env python3
"""Run the approved sole risk-distribution study from an already-written protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rmbfund.forecast_comparison import run_risk_distribution_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_risk_distribution_comparison(args.protocol, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return int(bool(result["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
