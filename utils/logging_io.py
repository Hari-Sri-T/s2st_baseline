"""
Logs one JSON record per pipeline run into results/metadata.json (a growing
list). This is what feeds the Baseline Failure Matrix in Week 2 — keep every
field even if it seems redundant now, you'll want it when comparing runs.
"""
import json
import os
from datetime import datetime, timezone


def append_result(results_dir: str, record: dict) -> None:
    meta_path = os.path.join(results_dir, "metadata.json")
    record = {**record, "timestamp": datetime.now(timezone.utc).isoformat()}

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
