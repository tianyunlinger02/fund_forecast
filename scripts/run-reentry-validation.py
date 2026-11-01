#!/usr/bin/env python3
"""Evaluate fixed post-sale reentry replacements on retained public NAV inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rmbfund.forecast_reentry_validation import run_reentry_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_reentry_validation(args.protocol, args.output)
    summary = {key: value for key, value in result.items() if key != "funds"}
    summary["funds"] = [{key: fund[key] for key in ("symbol", "status", "start_date", "end_date")}
                        for fund in result["funds"]]
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return int(bool(result["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
