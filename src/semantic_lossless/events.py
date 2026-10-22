"""Run-local events; a single collector appends deduplicated canonical CSV rows."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from .artifacts import clean_json

FIELDS = ["record_id", "timestamp", "algorithm", "variant", "run_id", "stage", "dataset", "split",
          "sample", "epoch", "step", "trial", "seed", "snr_db", "metric", "value", "unit", "status", "source"]


class Events:
    def __init__(self, run_dir, config):
        self.path = Path(run_dir) / "events.jsonl"
        self.identity = {"algorithm": config.algorithm, "variant": config.variant,
                         "run_id": Path(run_dir).name, "dataset": config.dataset,
                         "seed": config.seed, "snr_db": config.snr_db}

    def emit(self, stage, status="running", metrics=None, **details):
        record = clean_json({"record_id": uuid.uuid4().hex, "timestamp": datetime.now(timezone.utc).isoformat(),
                             **self.identity, "stage": stage, "status": status, "metrics": metrics or {}, **details})
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
        local_csv = self.path.parent / "metrics.csv"
        header = not local_csv.exists() or local_csv.stat().st_size == 0
        with local_csv.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if header:
                writer.writeheader()
            for name, value in (record.get("metrics") or {"stage_status": None}).items():
                row = {key: record.get(key, "") for key in FIELDS}
                row.update(record_id=f"{record['record_id']}:{name}", metric=name, value=value, source=self.path.as_posix())
                writer.writerow(row)
            stream.flush()
        print(json.dumps(record, allow_nan=False), flush=True)
        return record


def collect(root: Path, destination: Path) -> int:
    """Call from one controller; a lock rejects simultaneous CSV writers."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_suffix(destination.suffix + ".lock")
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError(f"Another collector owns {lock}; do not run two writers") from None
    try:
        handle.close()
        known = set()
        if destination.exists():
            with destination.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != FIELDS:
                    raise ValueError("Existing metrics.csv has a different schema; preserve it separately")
                known = {row["record_id"] for row in reader}
        rows = []
        for path in sorted(root.rglob("events.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a writer may not have finished its last line
                metrics = record.get("metrics") or {"stage_status": None}
                for name, value in metrics.items():
                    identifier = f"{record['record_id']}:{name}"
                    if identifier in known:
                        continue
                    row = {k: record.get(k, "") for k in FIELDS}
                    row.update(record_id=identifier, metric=name, value=value, source=path.as_posix())
                    rows.append(row)
                    known.add(identifier)
        if rows:
            header = not destination.exists() or destination.stat().st_size == 0
            with destination.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                if header:
                    writer.writeheader()
                writer.writerows(rows)
                stream.flush()
        return len(rows)
    finally:
        lock.unlink(missing_ok=True)
