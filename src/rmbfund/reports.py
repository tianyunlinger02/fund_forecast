"""Portable reports. Exports treat source text as untrusted, including spreadsheet formulas."""

from __future__ import annotations

import csv
import io
from typing import Any


def safe_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    stripped = text.lstrip(" \t\r\n﻿")
    return "'" + text if stripped.startswith(("=", "+", "-", "@")) else text


def csv_text(rows: list[dict[str, Any]]) -> str:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: safe_cell(value) for key, value in row.items()})
    return "﻿" + output.getvalue()
