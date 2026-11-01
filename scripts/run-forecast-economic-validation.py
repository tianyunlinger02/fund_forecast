#!/usr/bin/env python3
"""Run the declared forecast/policy ablation on retained public NAV inputs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rmbfund.forecast_economic_validation import run_economic_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_economic_validation(args.protocol, args.output)
    summary = {key: value for key, value in result.items() if key != "funds"}
    summary["funds"] = [{key: fund[key] for key in ("symbol", "status", "start_date", "end_date")}
                        for fund in result["funds"]]
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return int(bool(result["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
