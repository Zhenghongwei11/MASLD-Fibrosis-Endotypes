#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RE_FIB = re.compile(r"\b(fibrosis|fibrotic)\b", re.IGNORECASE)
RE_STAGE = re.compile(r"\b(stage|staging|fibrosis\s*stage)\b", re.IGNORECASE)


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def parse_characteristics(gz_path: Path) -> tuple[list[str], dict[str, list[str]]]:
    sample_ids: list[str] = []
    chars: dict[str, list[str]] = {}
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [_strip_quotes(p) for p in line.rstrip("\n").split("\t")[1:]]
                break
        if not sample_ids:
            raise RuntimeError(f"Missing !Sample_geo_accession in {gz_path}")
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_characteristics_ch1"):
                continue
            parts = [_strip_quotes(p) for p in line.rstrip("\n").split("\t")[1:]]
            if not parts:
                continue
            first = parts[0]
            if ":" not in first:
                continue
            key = first.split(":", 1)[0].strip()
            if not key:
                continue
            values: list[str] = []
            for p in parts:
                if ":" in p:
                    values.append(p.split(":", 1)[1].strip())
                else:
                    values.append("")
            if len(values) < len(sample_ids):
                values = values + [""] * (len(sample_ids) - len(values))
            values = values[: len(sample_ids)]
            chars[key] = values
    return sample_ids, chars


def _coerce_stage_value(raw: str) -> Optional[int]:
    s = (raw or "").strip()
    if not s:
        return None
    s = s.replace("–", "-")
    if "nash_f1_f4" in s.lower():
        return 3
    if "nash_f0" in s.lower():
        return 0
    m = re.search(r"\bF\s*([0-4])\b", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([0-4])\s*-\s*([0-4])\b", s)
    if m:
        return max(int(m.group(1)), int(m.group(2)))
    m = re.search(r"\bstage\s*([0-4])\b", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"[0-4]", s)
    if m:
        return int(s)
    return None


def choose_stage_key(keys: list[str]) -> str:
    fib_keys = [k for k in keys if RE_FIB.search(k) or RE_STAGE.search(k)]
    for k in fib_keys:
        if "fibrosis" in k.lower() and "stage" in k.lower():
            return k
    return fib_keys[0] if fib_keys else ""


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / "data" / "geo_cache"
    out_dir = repo_root / "results" / "endpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict[str, str]] = []
    dataset_rows: list[dict[str, str]] = []

    for gz in sorted(cache_dir.glob("*_series_matrix.txt.gz")):
        gse = gz.name.split("_", 1)[0]
        try:
            sample_ids, chars = parse_characteristics(gz)
        except Exception:
            continue

        keys = list(chars.keys())
        stage_key = choose_stage_key(keys)

        n_total = len(sample_ids)
        n_stage = 0
        n_f3plus = 0
        n_f4 = 0

        if stage_key:
            for sid, raw in zip(sample_ids, chars.get(stage_key, []), strict=True):
                st = _coerce_stage_value(raw)
                if st is None:
                    continue
                n_stage += 1
                n_f3plus += 1 if st >= 3 else 0
                n_f4 += 1 if st >= 4 else 0
                sample_rows.append(
                    {
                        "dataset_id": gse,
                        "sample_id": sid,
                        "fibrosis_stage": str(st),
                        "stage_key": stage_key,
                    }
                )

        dataset_rows.append(
            {
                "dataset_id": gse,
                "n_total": str(n_total),
                "stage_key": stage_key,
                "n_with_stage": str(n_stage),
                "n_f3plus": str(n_f3plus),
                "n_f4": str(n_f4),
            }
        )

    # Write sample endpoints
    sample_path = out_dir / "sample_endpoints.tsv"
    with sample_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=["dataset_id", "sample_id", "fibrosis_stage", "stage_key"])
        w.writeheader()
        w.writerows(sample_rows)

    summary_path = out_dir / "dataset_endpoints_summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["dataset_id", "n_total", "stage_key", "n_with_stage", "n_f3plus", "n_f4"],
        )
        w.writeheader()
        w.writerows(dataset_rows)

    print(str(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
