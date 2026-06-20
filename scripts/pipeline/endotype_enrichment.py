from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import hypergeom

from .expression_sources import load_expression_from_series_matrix_gene_symbols
from .multicohort_endotypes import DISCOVERY_GSE, select_variable_genes


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _read_gmt(path: Path, *, collection: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = sorted({g.strip().upper() for g in parts[2:] if g.strip()})
            if len(genes) < 5:
                continue
            rows.append({"collection": collection, "gene_set": name, "genes": genes})
    return rows


def _safe_ratio(num: int, den: int) -> float:
    return float(num / den) if den else np.nan


def run(repo_root: Path) -> None:
    out_dir = repo_root / "results" / "enrichment"
    _ensure_dirs(out_dir)

    sigs = pd.read_csv(repo_root / "results" / "endotypes" / "endotype_gene_signatures.tsv", sep="\t")
    sigs["gene"] = sigs["gene"].astype(str).str.upper()

    disc_expr = load_expression_from_series_matrix_gene_symbols(
        gse=DISCOVERY_GSE,
        cache_dir=repo_root / "data" / "geo_cache",
        probe_cache_dir=repo_root / "data" / "references",
        gene_subset=None,
    )
    universe = set(select_variable_genes(disc_expr.clip(lower=0), n=5000).index.astype(str).str.upper())
    M = len(universe)

    gene_sets: list[dict[str, Any]] = []
    reactome = repo_root / "data" / "references" / "gene_sets" / "Reactome_2022.gmt"
    hallmark = repo_root / "data" / "references" / "gene_sets" / "MSigDB_Hallmark_2020.gmt"
    if reactome.exists():
        gene_sets.extend(_read_gmt(reactome, collection="Reactome 2022"))
    if hallmark.exists():
        gene_sets.extend(_read_gmt(hallmark, collection="MSigDB Hallmark 2020"))
    if not gene_sets:
        raise FileNotFoundError("No GMT gene-set libraries found under data/references/gene_sets")

    rows: list[dict[str, Any]] = []
    for endotype, d in sigs.groupby("endotype"):
        query = set(d["gene"]).intersection(universe)
        N = len(query)
        for gs in gene_sets:
            gs_genes = set(gs["genes"]).intersection(universe)
            K = len(gs_genes)
            if K < 5 or N < 5:
                continue
            overlap = sorted(query.intersection(gs_genes))
            x = len(overlap)
            if x == 0:
                continue
            p = float(hypergeom.sf(x - 1, M, K, N))
            expected = N * K / M if M else np.nan
            rows.append(
                {
                    "endotype": endotype,
                    "collection": gs["collection"],
                    "gene_set": gs["gene_set"],
                    "overlap_size": x,
                    "signature_size_in_universe": N,
                    "gene_set_size_in_universe": K,
                    "universe_size": M,
                    "overlap_fraction": _safe_ratio(x, N),
                    "fold_enrichment": float(x / expected) if expected and expected > 0 else np.nan,
                    "pvalue": p,
                    "leading_genes": ",".join(overlap[:30]),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr"] = sm.stats.multipletests(out["pvalue"].to_numpy(float), method="fdr_bh")[1]
        out = out.sort_values(["endotype", "fdr", "pvalue", "gene_set"]).reset_index(drop=True)
    out.to_csv(out_dir / "endotype_pathway_enrichment.tsv", sep="\t", index=False)

