#!/usr/bin/env python3
"""Evaluate fixed buy/hold/sell source consistency on retained public histories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rmbfund.forecast_consistency_validation import run_consistency_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_consistency_validation(args.protocol, args.output)
    summary = {key: value for key, value in result.items() if key not in {"funds", "control_reproduction"}}
    summary["control_reproduction_count"] = len(result["control_reproduction"])
    summary["funds"] = [{key: fund[key] for key in ("symbol", "status", "start_date", "end_date")}
                        for fund in result["funds"]]
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
