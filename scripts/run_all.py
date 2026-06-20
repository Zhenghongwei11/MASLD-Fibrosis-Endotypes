#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import subprocess
from pathlib import Path

import pandas as pd

from pipeline.dataset_summary import build_dataset_summary
from pipeline.geo_download import ensure_series_matrix_cached
from pipeline.multicohort_endotypes import run as run_multicohort
from pipeline.scrna_localization import run as run_scrna_localization
from pipeline.signature_benchmarking import run as run_signature_benchmarking
from pipeline.endotype_enrichment import run as run_endotype_enrichment
from pipeline.discovery_sensitivity import run as run_discovery_sensitivity
from pipeline.plots import (
    plot_cohort_landscape,
    plot_forest_meta,
    plot_loco_performance,
    plot_prediction_srt_style,
    plot_sample_flow_diagram,
    plot_study_design_schematic,
    plot_scrna_localization,
    plot_signature_coverage,
    plot_signature_benchmarking,
    plot_pathway_annotation,
    plot_robustness_sensitivity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _now_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_audit(run_id: str, payload: dict) -> None:
    out_dir = REPO_ROOT / "docs" / "audit_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stage_results(run_id: str) -> None:
    # 0) Ensure required GEO series matrices are cached (endpoint extraction relies on them).
    geo_cache = REPO_ROOT / "data" / "geo_cache"
    for gse in ["GSE163211", "GSE48452", "GSE49541", "GSE130970", "GSE162694", "GSE135251"]:
        try:
            ensure_series_matrix_cached(gse=gse, cache_dir=geo_cache)
        except Exception:
            # Non-fatal: downstream steps will skip missing cohorts defensively.
            pass

    # 1) Dataset summary
    build_dataset_summary(REPO_ROOT / "data" / "geo_cache", REPO_ROOT / "results" / "dataset_summary.tsv")

    # 2) Endpoint harmonization (fibrosis staging extraction)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "metadata" / "build_endpoint_harmonization_masld.py")],
        check=True,
    )

    # 3) Multi-cohort analysis + meta-analysis + LOCO evaluation
    run_multicohort(REPO_ROOT)

    # 3b) Orthogonal evidence (optional): liver scRNA cell-type localization
    run_scrna_localization(REPO_ROOT, allow_download=True)

    # 4) BMC Genomics upgrade analyses: prior signatures, pathway annotation,
    # and discovery-cohort dependence checks.
    run_signature_benchmarking(REPO_ROOT)
    run_endotype_enrichment(REPO_ROOT)
    run_discovery_sensitivity(REPO_ROOT)

    # 5) Robustness grid (fast enough for local full runs; also used for Figure 5)
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "pipeline" / "robustness_gse163211.py")], check=True)

    _write_audit(run_id, {"stage": "results", "ok": True})


def stage_figures(run_id: str) -> None:
    out_dir = REPO_ROOT / "plots" / "publication"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_study_design_schematic(out_dir)
    plot_cohort_landscape(REPO_ROOT / "results" / "figures" / "cohort_landscape.tsv", out_dir)
    plot_sample_flow_diagram(
        REPO_ROOT / "results" / "endpoints" / "dataset_endpoints_summary.tsv",
        REPO_ROOT / "results" / "figures" / "signature_coverage.tsv",
        out_dir,
    )
    plot_forest_meta(
        REPO_ROOT / "results" / "endotypes" / "endotype_fibrosis_effects_multicohort.tsv",
        REPO_ROOT / "results" / "effect_sizes" / "meta_analysis_endotype_f3plus.tsv",
        out_dir,
    )
    plot_loco_performance(REPO_ROOT / "results" / "benchmarks" / "loco_predictions.tsv", out_dir)
    plot_signature_coverage(REPO_ROOT / "results" / "figures" / "signature_coverage.tsv", out_dir)
    plot_signature_benchmarking(
        REPO_ROOT / "results" / "effect_sizes" / "meta_analysis_endotype_f3plus.tsv",
        REPO_ROOT / "results" / "benchmarking" / "prior_signature_meta_analysis.tsv",
        REPO_ROOT / "results" / "benchmarking" / "endotype_prior_signature_correlations.tsv",
        out_dir,
    )
    plot_pathway_annotation(REPO_ROOT / "results" / "enrichment" / "endotype_pathway_enrichment.tsv", out_dir)
    plot_robustness_sensitivity(REPO_ROOT / "results" / "robustness" / "nmf_sensitivity.tsv", out_dir)
    plot_prediction_srt_style(REPO_ROOT / "results" / "benchmarks" / "loco_predictions.tsv", out_dir)
    plot_scrna_localization(
        REPO_ROOT / "results" / "scrna" / "GSE115469_endotype_celltype_localization.tsv",
        out_dir,
    )
    _write_audit(run_id, {"stage": "figures", "ok": True})


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end pipeline runner (data -> results -> plots).")
    ap.add_argument("--run-id", default=_now_run_id(), help="Run identifier for audit bundle output.")
    ap.add_argument("--stage", choices=["results", "figures", "all"], default="all")
    args = ap.parse_args()

    if args.stage in {"results", "all"}:
        stage_results(args.run_id)
    if args.stage in {"figures", "all"}:
        stage_figures(args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
