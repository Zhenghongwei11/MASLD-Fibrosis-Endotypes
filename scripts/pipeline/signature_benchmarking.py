from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from docx import Document
from scipy import stats
from sklearn.metrics import roc_auc_score

from .expression_sources import (
    get_default_expression_cohorts,
    load_expression_from_series_matrix_gene_symbols,
    load_expression_gse130970,
    load_expression_gse135251,
    load_expression_gse162694,
)
from .gene_id_map import load_gene_id_maps
from .geo_series_matrix import read_series_matrix_header
from .multicohort_endotypes import _glm_or_per_1sd, dersimonian_laird


HE_2023_DOI = "10.1186/s12967-023-04300-6"


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _split_genes(text: str) -> list[str]:
    genes: list[str] = []
    for token in re.split(r"[,;\s]+", text or ""):
        g = token.strip().upper()
        if re.match(r"^[A-Z0-9][A-Z0-9.-]*$", g):
            genes.append(g)
    return list(dict.fromkeys(genes))


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0, np.nan)
    return (df.sub(mu, axis=0)).div(sd, axis=0)


def _standardize_col(s: pd.Series) -> pd.Series:
    sd = float(s.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return s * np.nan
    return (s - float(s.mean())) / sd


def extract_he_2023_signatures(docx_path: Path, out_tsv: Path) -> pd.DataFrame:
    doc = Document(docx_path)
    rows: list[dict[str, Any]] = []
    wanted_tables = {
        0: "Additional file 2 Table S1",
        2: "Additional file 2 Table S3",
    }
    for table_idx, source_table in wanted_tables.items():
        table = doc.tables[table_idx]
        for r in table.rows[1:]:
            raw_name = r.cells[0].text.strip()
            genes = _split_genes(r.cells[1].text)
            if not raw_name or not genes:
                continue
            name = re.sub(r"\s+", " ", raw_name).strip()
            name = {
                "LiverAgingsignature": "Liver aging signature",
                "HSCTGFbetasignature": "HSC TGF-beta signature",
                "CD4 Tcells": "CD4 T-cell signature",
                "CD8 Tcells": "CD8 T-cell signature",
            }.get(name, name)
            for rank, gene in enumerate(genes, start=1):
                rows.append(
                    {
                        "signature": name,
                        "rank": rank,
                        "gene": gene,
                        "source": "He et al. 2023",
                        "doi": HE_2023_DOI,
                        "source_table": source_table,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(out_tsv, sep="\t", index=False)
    return out


def _load_expr_for_cohort(
    *,
    gse: str,
    repo_root: Path,
    gene_subset: set[str],
) -> tuple[pd.DataFrame, str]:
    geo_cache_dir = repo_root / "data" / "geo_cache"
    suppl_cache_dir = repo_root / "data" / "geo_suppl"
    ref_cache_dir = repo_root / "data" / "references"
    gene_maps = load_gene_id_maps(ref_cache_dir)
    platform_id = ""
    gz = geo_cache_dir / f"{gse}_series_matrix.txt.gz"
    if gz.exists():
        platform_id, _, _ = read_series_matrix_header(gz)

    if gse == "GSE130970":
        expr = load_expression_gse130970(geo_cache_dir=geo_cache_dir, suppl_cache_dir=suppl_cache_dir, gene_maps=gene_maps)
        expr = expr.loc[expr.index.intersection(sorted(gene_subset))]
    elif gse == "GSE162694":
        expr = load_expression_gse162694(geo_cache_dir=geo_cache_dir, suppl_cache_dir=suppl_cache_dir, gene_maps=gene_maps)
        expr = expr.loc[expr.index.intersection(sorted(gene_subset))]
    elif gse == "GSE135251":
        expr = load_expression_gse135251(suppl_cache_dir=suppl_cache_dir, gene_maps=gene_maps, gene_subset=gene_subset)
    else:
        expr = load_expression_from_series_matrix_gene_symbols(
            gse=gse,
            cache_dir=geo_cache_dir,
            probe_cache_dir=ref_cache_dir,
            gene_subset=gene_subset,
        )
    expr.index = expr.index.astype(str).str.upper()
    expr = expr.groupby(level=0).mean()
    return expr, platform_id


def run(repo_root: Path, *, cohorts: list[str] | None = None) -> None:
    out_dir = repo_root / "results" / "benchmarking"
    _ensure_dirs(out_dir)

    docx_path = repo_root / "data" / "references" / "prior_signatures" / "he_2023_additional_file_2.docx"
    signature_tsv = out_dir / "prior_signature_definitions.tsv"
    sig_defs = extract_he_2023_signatures(docx_path, signature_tsv)

    signatures = {
        name: g["gene"].astype(str).str.upper().tolist()
        for name, g in sig_defs.groupby("signature", sort=False)
    }
    gene_union = set(sig_defs["gene"].astype(str).str.upper())

    endpoints = pd.read_csv(repo_root / "results" / "endpoints" / "sample_endpoints.tsv", sep="\t")
    endpoints["fibrosis_stage"] = pd.to_numeric(endpoints["fibrosis_stage"], errors="coerce")
    endpoints = endpoints.dropna(subset=["fibrosis_stage"])
    endpoints["y_f3plus"] = (endpoints["fibrosis_stage"] >= 3).astype(int)
    if cohorts is None:
        cohorts = get_default_expression_cohorts()

    score_rows: list[dict[str, Any]] = []
    assoc_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for gse in cohorts:
        ep = endpoints[endpoints["dataset_id"] == gse].copy()
        if ep.empty:
            continue
        sample_ids = set(ep["sample_id"].astype(str))
        try:
            expr, platform_id = _load_expr_for_cohort(gse=gse, repo_root=repo_root, gene_subset=gene_union)
        except Exception:
            continue
        keep_cols = [c for c in expr.columns if str(c) in sample_ids]
        expr = expr[keep_cols]
        if expr.shape[1] < 20:
            continue
        y = ep.set_index("sample_id")["y_f3plus"].reindex([str(c) for c in expr.columns])
        if y.dropna().shape[0] < 20 or y.nunique(dropna=True) < 2:
            continue

        z = _zscore_rows(expr)
        scores = pd.DataFrame(index=expr.columns)
        for name, genes in signatures.items():
            avail = [g for g in genes if g in z.index]
            coverage_rows.append(
                {
                    "dataset_id": gse,
                    "platform_id": platform_id,
                    "signature": name,
                    "n_sig_genes_used": len(avail),
                    "n_sig_genes_total": len(genes),
                    "coverage": len(avail) / len(genes) if genes else np.nan,
                }
            )
            if len(avail) < max(5, int(0.20 * len(genes))):
                scores[name] = np.nan
            else:
                scores[name] = z.loc[avail].mean(axis=0)
        scores = scores.apply(_standardize_col, axis=0)

        for sid, row in scores.iterrows():
            for name, value in row.items():
                score_rows.append(
                    {
                        "dataset_id": gse,
                        "sample_id": str(sid),
                        "signature": name,
                        "score": float(value) if pd.notna(value) else np.nan,
                    }
                )

        for name in scores.columns:
            res = _glm_or_per_1sd(y, scores[name])
            if res is None:
                continue
            d = pd.concat([y.rename("y"), scores[name].rename("score")], axis=1).dropna()
            auc = np.nan
            if d["y"].nunique() == 2 and d["score"].notna().sum() >= 20:
                try:
                    auc = float(roc_auc_score(d["y"].to_numpy(dtype=int), d["score"].to_numpy(dtype=float)))
                except Exception:
                    auc = np.nan
            assoc_rows.append(
                {
                    "dataset_id": gse,
                    "platform_id": platform_id,
                    "signature": name,
                    "source": "He et al. 2023",
                    "doi": HE_2023_DOI,
                    "outcome": "advanced_fibrosis_F3plus_vs_F0F2",
                    "effect_type": "OR_per_1SD",
                    "effect": res["or"],
                    "ci_lower": res["ci_lower"],
                    "ci_upper": res["ci_upper"],
                    "pvalue": res["pvalue"],
                    "log_or": res["log_or"],
                    "se": res["se"],
                    "auc": auc,
                    "n": res["n"],
                    "events": res["events"],
                }
            )

    assoc = pd.DataFrame(assoc_rows)
    if not assoc.empty:
        assoc["fdr"] = sm.stats.multipletests(assoc["pvalue"].to_numpy(dtype=float), method="fdr_bh")[1]
    assoc.to_csv(out_dir / "prior_signature_associations.tsv", sep="\t", index=False)
    pd.DataFrame(score_rows).to_csv(out_dir / "prior_signature_scores.tsv", sep="\t", index=False)
    pd.DataFrame(coverage_rows).to_csv(out_dir / "prior_signature_coverage.tsv", sep="\t", index=False)

    # Correlate prior signatures with the transferred Endotype 1 score in the same samples.
    prior_scores = pd.DataFrame(score_rows)
    endotype_scores = pd.read_csv(repo_root / "results" / "figures" / "endotype_scores_multicohort.tsv", sep="\t")
    corr_rows: list[dict[str, Any]] = []
    if not prior_scores.empty and not endotype_scores.empty:
        e1 = endotype_scores[["dataset_id", "sample_id", "endotype_1"]].copy()
        e1["sample_id"] = e1["sample_id"].astype(str)
        for (gse, sig), d in prior_scores.groupby(["dataset_id", "signature"]):
            merged = d.merge(e1, on=["dataset_id", "sample_id"], how="inner").dropna(subset=["score", "endotype_1"])
            if merged.shape[0] < 10:
                continue
            r, p = stats.pearsonr(merged["score"].to_numpy(float), merged["endotype_1"].to_numpy(float))
            rho, sp = stats.spearmanr(merged["score"].to_numpy(float), merged["endotype_1"].to_numpy(float))
            corr_rows.append(
                {
                    "dataset_id": gse,
                    "signature": sig,
                    "comparison": "prior_signature_vs_endotype_1",
                    "pearson_r": float(r),
                    "pearson_pvalue": float(p),
                    "spearman_rho": float(rho),
                    "spearman_pvalue": float(sp),
                    "n": int(merged.shape[0]),
                }
            )
    corr = pd.DataFrame(corr_rows)
    if not corr.empty:
        corr["pearson_fdr"] = sm.stats.multipletests(corr["pearson_pvalue"].to_numpy(float), method="fdr_bh")[1]
        corr["spearman_fdr"] = sm.stats.multipletests(corr["spearman_pvalue"].to_numpy(float), method="fdr_bh")[1]
    corr.to_csv(out_dir / "endotype_prior_signature_correlations.tsv", sep="\t", index=False)

    meta_rows: list[dict[str, Any]] = []
    if not assoc.empty:
        for sig, d in assoc.dropna(subset=["log_or", "se"]).groupby("signature"):
            if d.shape[0] < 2:
                continue
            m = dersimonian_laird(d["log_or"].to_numpy(float), d["se"].to_numpy(float))
            meta_rows.append(
                {
                    "signature": sig,
                    "source": "He et al. 2023",
                    "doi": HE_2023_DOI,
                    "k": int(m["k"]),
                    "or_pooled": math.exp(m["mu_log_or"]),
                    "ci_lower": math.exp(m["ci_lower_log"]),
                    "ci_upper": math.exp(m["ci_upper_log"]),
                    "prediction_interval_lower": math.exp(m["pi_lower_log"]),
                    "prediction_interval_upper": math.exp(m["pi_upper_log"]),
                    "I2_percent": m["I2_percent"],
                    "tau2": m["tau2"],
                    "median_auc": float(d["auc"].median(skipna=True)),
                }
            )
    pd.DataFrame(meta_rows).to_csv(out_dir / "prior_signature_meta_analysis.tsv", sep="\t", index=False)
