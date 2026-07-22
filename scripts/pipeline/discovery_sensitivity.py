from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import hypergeom
from scipy import stats
from sklearn.decomposition import NMF

from .expression_sources import load_expression_gse135251
from .gene_id_map import load_gene_id_maps
from .multicohort_endotypes import _glm_or_per_1sd, select_variable_genes


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0, np.nan)
    return (df.sub(mu, axis=0)).div(sd, axis=0)


def _standardize_col(s: pd.Series) -> pd.Series:
    sd = float(s.std(ddof=0))
    return (s - float(s.mean())) / sd if np.isfinite(sd) and sd != 0 else s * np.nan


def _fit_nmf_loadings(expr_log: pd.DataFrame, *, k: int = 3, seed: int = 1) -> pd.DataFrame:
    model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=2000)
    H = model.fit_transform(expr_log.T.to_numpy(dtype=float))
    # We need component loadings, not sample scores.
    # sklearn stores gene loadings as components_ after fit_transform.
    loadings = model.components_
    return pd.DataFrame(loadings, index=[f"gse135251_component_{i+1}" for i in range(k)], columns=expr_log.index)


def run(repo_root: Path, *, k: int = 3, seed: int = 1, topk: int = 60) -> None:
    out_dir = repo_root / "results" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    endotype_defs = pd.read_csv(repo_root / "results" / "endotypes" / "endotype_gene_signatures.tsv", sep="\t")
    endotype_defs["gene"] = endotype_defs["gene"].astype(str).str.upper()
    current = {
        e: set(d["gene"].head(topk))
        for e, d in endotype_defs.sort_values(["endotype", "rank"]).groupby("endotype")
    }

    gene_maps = load_gene_id_maps(repo_root / "data" / "references")
    expr = load_expression_gse135251(
        suppl_cache_dir=repo_root / "data" / "geo_suppl",
        gene_maps=gene_maps,
        gene_subset=None,
    ).clip(lower=0)
    expr.index = expr.index.astype(str).str.upper()
    expr = expr.groupby(level=0).mean()
    expr_log = select_variable_genes(expr, n=5000)
    H = _fit_nmf_loadings(expr_log, k=k, seed=seed)
    universe = set(expr_log.index.astype(str).str.upper())
    M = len(universe)

    endpoints = pd.read_csv(repo_root / "results" / "endpoints" / "sample_endpoints.tsv", sep="\t")
    endpoints = endpoints[endpoints["dataset_id"] == "GSE135251"].copy()
    endpoints["fibrosis_stage"] = pd.to_numeric(endpoints["fibrosis_stage"], errors="coerce")
    endpoints = endpoints.dropna(subset=["fibrosis_stage"])
    endpoints["y_f3plus"] = (endpoints["fibrosis_stage"] >= 3).astype(int)
    y = endpoints.set_index("sample_id")["y_f3plus"]
    primary_scores = pd.read_csv(repo_root / "results" / "figures" / "endotype_scores_multicohort.tsv", sep="\t")
    primary_scores = primary_scores[primary_scores["dataset_id"] == "GSE135251"].copy()
    primary_scores["sample_id"] = primary_scores["sample_id"].astype(str)

    rows: list[dict[str, Any]] = []
    alt_defs: list[dict[str, Any]] = []
    z = _zscore_rows(expr)
    for comp in H.index:
        alt_genes = [g.upper() for g in H.loc[comp].sort_values(ascending=False).head(topk).index.astype(str)]
        alt_set = set(alt_genes)
        for rank, gene in enumerate(alt_genes, start=1):
            alt_defs.append({"alternate_discovery": "GSE135251", "component": comp, "rank": rank, "gene": gene})

        avail = [g for g in alt_genes if g in z.index]
        score = z.loc[avail].mean(axis=0) if avail else pd.Series(index=expr.columns, dtype=float)
        score = _standardize_col(score)
        y2 = y.reindex([str(c) for c in score.index])
        assoc = _glm_or_per_1sd(y2, score)
        alt_score = pd.DataFrame({"sample_id": score.index.astype(str), "alternate_score": score.to_numpy(float)})

        for endotype, genes in current.items():
            e_col = endotype
            score_corr = np.nan
            score_corr_p = np.nan
            if e_col in primary_scores.columns:
                merged_scores = primary_scores[["sample_id", e_col]].merge(alt_score, on="sample_id", how="inner").dropna()
                if merged_scores.shape[0] >= 10:
                    r, p_corr = stats.pearsonr(
                        merged_scores[e_col].to_numpy(float),
                        merged_scores["alternate_score"].to_numpy(float),
                    )
                    score_corr = float(r)
                    score_corr_p = float(p_corr)
            genes_in_universe = set(genes).intersection(universe)
            alt_in_universe = alt_set.intersection(universe)
            overlap = sorted(genes_in_universe.intersection(alt_in_universe))
            x = len(overlap)
            K = len(genes_in_universe)
            N = len(alt_in_universe)
            p = float(hypergeom.sf(x - 1, M, K, N)) if x > 0 and M and K and N else 1.0
            rows.append(
                {
                    "primary_discovery": "GSE163211",
                    "alternate_discovery": "GSE135251",
                    "alternate_component": comp,
                    "primary_endotype": endotype,
                    "topk": topk,
                    "overlap_size": x,
                    "jaccard": x / len(genes_in_universe.union(alt_in_universe)) if genes_in_universe.union(alt_in_universe) else np.nan,
                    "hypergeom_pvalue": p,
                    "overlap_genes": ",".join(overlap),
                    "score_pearson_r": score_corr,
                    "score_pearson_pvalue": score_corr_p,
                    "alternate_component_or": assoc["or"] if assoc else np.nan,
                    "alternate_component_ci_lower": assoc["ci_lower"] if assoc else np.nan,
                    "alternate_component_ci_upper": assoc["ci_upper"] if assoc else np.nan,
                    "alternate_component_pvalue": assoc["pvalue"] if assoc else np.nan,
                    "alternate_component_n": assoc["n"] if assoc else np.nan,
                    "alternate_component_events": assoc["events"] if assoc else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["hypergeom_fdr"] = sm.stats.multipletests(out["hypergeom_pvalue"].to_numpy(float), method="fdr_bh")[1]
        out = out.sort_values(["primary_endotype", "hypergeom_fdr", "alternate_component"]).reset_index(drop=True)
    out.to_csv(out_dir / "discovery_sensitivity.tsv", sep="\t", index=False)
    pd.DataFrame(alt_defs).to_csv(out_dir / "gse135251_nmf_gene_signatures.tsv", sep="\t", index=False)
