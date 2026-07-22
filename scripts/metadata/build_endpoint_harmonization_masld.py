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


@dataclass(frozen=True)
class StageParse:
    stage: int | None
    eligible_f3plus: bool
    status: str
    reason: str


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
            if len(parts) < len(sample_ids):
                parts = parts + [""] * (len(sample_ids) - len(parts))
            parts = parts[: len(sample_ids)]

            # GEO characteristics rows are usually row-homogeneous, but some
            # series mix different keys across samples in the same row. Parse
            # each cell independently to avoid shifting fibrosis labels into
            # unrelated fields.
            for idx, p in enumerate(parts):
                if ":" not in p:
                    continue
                key, value = p.split(":", 1)
                key = key.strip()
                if not key:
                    continue
                chars.setdefault(key, [""] * len(sample_ids))
                chars[key][idx] = value.strip()
    return sample_ids, chars


def _coerce_stage_value(raw: str) -> Optional[int]:
    parsed = parse_stage_value(raw)
    return parsed.stage if parsed.eligible_f3plus else None


def parse_stage_value(raw: str) -> StageParse:
    s = (raw or "").strip()
    if not s:
        return StageParse(None, False, "missing", "empty")
    s = s.replace("–", "-")
    s_norm = s.lower().replace("_", " ")
    if "nash f1 f4" in s_norm or re.search(r"\bf\s*1\s*-\s*f?\s*4\b", s_norm):
        return StageParse(None, False, "ambiguous_range", "range_crosses_f3_threshold")
    if "nash_f0" in s.lower():
        return StageParse(0, True, "explicit_stage", "")
    m = re.search(r"\bF\s*([0-4])\b", s, flags=re.IGNORECASE)
    if m:
        return StageParse(int(m.group(1)), True, "explicit_stage", "")
    m = re.search(r"\b([0-4])\s*-\s*([0-4])\b", s)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2))
        if lo == hi:
            return StageParse(lo, True, "explicit_stage", "")
        if hi <= 2:
            return StageParse(hi, True, "range_within_nonadvanced", "range_within_f0_f2")
        if lo >= 3:
            return StageParse(hi, True, "range_within_advanced", "range_within_f3_f4")
        return StageParse(None, False, "ambiguous_range", "range_crosses_f3_threshold")
    m = re.search(r"\bstage\s*([0-4])\b", s, flags=re.IGNORECASE)
    if m:
        return StageParse(int(m.group(1)), True, "explicit_stage", "")
    m = re.fullmatch(r"[0-4]", s)
    if m:
        return StageParse(int(s), True, "explicit_stage", "")
    return StageParse(None, False, "unparsed", "no_supported_stage_pattern")


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
        f4_not_separately_available = False

        if stage_key:
            for sid, raw in zip(sample_ids, chars.get(stage_key, []), strict=True):
                parsed = parse_stage_value(raw)
                st = parsed.stage
                if parsed.eligible_f3plus and st is not None:
                    n_stage += 1
                    n_f3plus += 1 if st >= 3 else 0
                    if parsed.status == "explicit_stage" and st >= 4:
                        n_f4 += 1
                    elif parsed.status == "range_within_advanced" and st >= 4:
                        f4_not_separately_available = True
                sample_rows.append(
                    {
                        "dataset_id": gse,
                        "sample_id": sid,
                        "fibrosis_stage": "" if st is None else str(st),
                        "stage_key": stage_key,
                        "stage_raw": raw,
                        "eligible_f3plus": "1" if parsed.eligible_f3plus else "0",
                        "parse_status": parsed.status,
                        "exclusion_reason": parsed.reason,
                    }
                )

        dataset_rows.append(
            {
                "dataset_id": gse,
                "n_total": str(n_total),
                "stage_key": stage_key,
                "n_with_stage": str(n_stage),
                "n_f3plus": str(n_f3plus),
                "n_f4": "not_separately_available" if f4_not_separately_available else str(n_f4),
            }
        )

    # Write sample endpoints
    sample_path = out_dir / "sample_endpoints.tsv"
    with sample_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "dataset_id",
                "sample_id",
                "fibrosis_stage",
                "stage_key",
                "stage_raw",
                "eligible_f3plus",
                "parse_status",
                "exclusion_reason",
            ],
        )
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
