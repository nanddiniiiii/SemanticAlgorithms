"""CSV output without a pandas dependency."""

import csv
from pathlib import Path
from typing import Any

from Evaluation.Reporting.comparison_runner import validate_comparison_row


def write_result_table(rows: list[dict[str, Any]], path: str | Path) -> Path:
    if not rows:
        raise ValueError("Cannot create a result table without rows.")
    for row in rows:
        validate_comparison_row(row)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target
