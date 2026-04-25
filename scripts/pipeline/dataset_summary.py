from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .geo_series_matrix import has_expression_table, read_series_matrix_header


def build_dataset_summary(cache_dir: Path, out_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gz in sorted(cache_dir.glob("*_series_matrix.txt.gz")):
        gse = gz.name.split("_", 1)[0]
        platform_id, sample_ids, characteristics = read_series_matrix_header(gz)
        row: Dict[str, Any] = {
            "dataset_id": gse,
            "platform_id": platform_id,
            "n_samples": len([s for s in sample_ids if s]),
            "expression_table_present": bool(has_expression_table(gz)),
        }
        keys = sorted(characteristics.keys())
        row["characteristic_keys"] = "|".join(keys[:40]) + ("|..." if len(keys) > 40 else "")
        if "nafld stage" in characteristics:
            levels = sorted({v for v in characteristics["nafld stage"] if v})
            row["nafld_stage_levels"] = "|".join(levels)
        out = pd.DataFrame([row])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["expression_table_present", "n_samples"], ascending=[False, False])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    return df

