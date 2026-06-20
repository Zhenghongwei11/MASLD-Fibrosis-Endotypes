from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .expression_sources import (
    get_default_expression_cohorts,
    load_expression_from_series_matrix_gene_symbols,
    load_expression_gse130970,
    load_expression_gse135251,
    load_expression_gse162694,
)
from .gene_id_map import load_gene_id_maps
from .geo_series_matrix import read_series_matrix_header


DISCOVERY_GSE = "GSE163211"


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=0)
    sd = df.std(axis=0, ddof=0).replace(0, np.nan)
    return (df - mu) / sd


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0, np.nan)
    return (df.sub(mu, axis=0)).div(sd, axis=0)


def select_variable_genes(expr: pd.DataFrame, n: int = 5000) -> pd.DataFrame:
    x = np.log1p(expr.clip(lower=0))
    var = x.var(axis=1)
    keep = var.sort_values(ascending=False).head(n).index
    return x.loc[keep]


def fit_nmf(expr_log: pd.DataFrame, k: int = 3, seed: int = 1) -> pd.DataFrame:
    X = expr_log.T.to_numpy(dtype=float)
    model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=2000)
    W = model.fit_transform(X)  # samples x k
    H = model.components_  # k x genes
    H_df = pd.DataFrame(H, index=[f"endotype_{i+1}" for i in range(k)], columns=expr_log.index)
    return H_df


def build_signatures(H: pd.DataFrame, topk: int = 60) -> dict[str, list[str]]:
    sigs: dict[str, list[str]] = {}
    for e in H.index:
        genes = H.loc[e].sort_values(ascending=False).head(topk).index.astype(str).tolist()
        sigs[e] = genes
    return sigs


def _glm_or_per_1sd(y: pd.Series, x: pd.Series) -> dict[str, float] | None:
    data = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if data.empty or data["y"].nunique(dropna=True) < 2:
        return None
    model = sm.GLM(data["y"], sm.add_constant(data["x"], has_constant="add"), family=sm.families.Binomial())
    try:
        fit = model.fit()
    except Exception:
        return None
    coef = float(fit.params["x"])
    se = float(fit.bse["x"])
    or_ = math.exp(coef)
    ci_lo = math.exp(coef - 1.96 * se)
    ci_hi = math.exp(coef + 1.96 * se)
    p = float(fit.pvalues["x"])
    return {
        "log_or": coef,
        "se": se,
        "or": or_,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "pvalue": p,
        "n": int(data.shape[0]),
        "events": int(data["y"].sum()),
    }


def dersimonian_laird(theta: np.ndarray, se: np.ndarray) -> dict[str, float]:
    w = 1.0 / np.square(se)
    mu_fixed = float(np.sum(w * theta) / np.sum(w))
    Q = float(np.sum(w * np.square(theta - mu_fixed)))
    df = max(1, int(len(theta) - 1))
    C = float(np.sum(w) - (np.sum(np.square(w)) / np.sum(w)))
    tau2 = 0.0 if C <= 0 else max(0.0, (Q - df) / C)
    w_re = 1.0 / (np.square(se) + tau2)
    mu_re = float(np.sum(w_re * theta) / np.sum(w_re))
    se_re = float(math.sqrt(1.0 / np.sum(w_re)))
    ci_lo = mu_re - 1.96 * se_re
    ci_hi = mu_re + 1.96 * se_re
    I2 = 0.0
    if Q > df and Q > 0:
        I2 = max(0.0, (Q - df) / Q) * 100.0
    pred_sd = float(math.sqrt(tau2 + se_re * se_re))
    pi_lo = mu_re - 1.96 * pred_sd
    pi_hi = mu_re + 1.96 * pred_sd
    return {
        "k": float(len(theta)),
        "mu_log_or": mu_re,
        "se_mu": se_re,
        "ci_lower_log": ci_lo,
        "ci_upper_log": ci_hi,
        "tau2": tau2,
        "I2_percent": I2,
        "Q": Q,
        "pi_lower_log": pi_lo,
        "pi_upper_log": pi_hi,
    }


def calibration_slope(y_true: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    fit = sm.GLM(y_true, sm.add_constant(logit, has_constant="add"), family=sm.families.Binomial()).fit()
    return float(fit.params[1])


def run(
    repo_root: Path,
    *,
    cohorts: list[str] | None = None,
    seed: int = 1,
    k: int = 3,
    topk: int = 60,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(repo_root.resolve()))

    geo_cache_dir = repo_root / "data" / "geo_cache"
    suppl_cache_dir = repo_root / "data" / "geo_suppl"
    ref_cache_dir = repo_root / "data" / "references"
    probe_cache_dir = ref_cache_dir
    gene_maps = load_gene_id_maps(ref_cache_dir)

    endpoints_path = repo_root / "results" / "endpoints" / "sample_endpoints.tsv"
    endpoints = pd.read_csv(endpoints_path, sep="\t")
    endpoints["fibrosis_stage"] = pd.to_numeric(endpoints["fibrosis_stage"], errors="coerce")
    endpoints = endpoints.dropna(subset=["fibrosis_stage"])
    endpoints["y_f3plus"] = (endpoints["fibrosis_stage"] >= 3).astype(int)

    if cohorts is None:
        cohorts = get_default_expression_cohorts()

    # Discovery: fit NMF and define signatures
    disc_expr = load_expression_from_series_matrix_gene_symbols(
        gse=DISCOVERY_GSE,
        cache_dir=geo_cache_dir,
        probe_cache_dir=probe_cache_dir,
        gene_subset=None,
    )
    disc_expr = disc_expr.clip(lower=0)
    disc_log = select_variable_genes(disc_expr, n=5000)
    H = fit_nmf(disc_log, k=k, seed=seed)
    sigs = build_signatures(H, topk=topk)
    sig_union = set(sum(sigs.values(), []))

    out_results = repo_root / "results"
    endotypes_dir = out_results / "endotypes"
    effect_dir = out_results / "effect_sizes"
    bench_dir = out_results / "benchmarks"
    figures_dir = out_results / "figures"
    _ensure_dirs(endotypes_dir, effect_dir, bench_dir, figures_dir)

    # Save signatures
    defs = []
    for e, genes in sigs.items():
        for rank, gene in enumerate(genes, start=1):
            defs.append({"endotype": e, "rank": rank, "gene": gene})
    pd.DataFrame(defs).to_csv(endotypes_dir / "endotype_gene_signatures.tsv", sep="\t", index=False)

    cohort_rows = []
    assoc_rows = []
    score_rows = []
    coverage_rows = []

    # Load cohorts and compute association
    for gse in cohorts:
        ep = endpoints[endpoints["dataset_id"] == gse].copy()
        if ep.empty:
            continue
        sample_ids = ep["sample_id"].astype(str).tolist()
        y = ep.set_index("sample_id")["y_f3plus"].rename("y_f3plus")

        platform_id = ""
        gz = geo_cache_dir / f"{gse}_series_matrix.txt.gz"
        if gz.exists():
            platform_id, _, _ = read_series_matrix_header(gz)

        try:
            if gse in {"GSE130970"}:
                expr = load_expression_gse130970(
                    geo_cache_dir=geo_cache_dir, suppl_cache_dir=suppl_cache_dir, gene_maps=gene_maps
                )
            elif gse in {"GSE162694"}:
                expr = load_expression_gse162694(
                    geo_cache_dir=geo_cache_dir, suppl_cache_dir=suppl_cache_dir, gene_maps=gene_maps
                )
            elif gse in {"GSE135251"}:
                expr = load_expression_gse135251(
                    suppl_cache_dir=suppl_cache_dir, gene_maps=gene_maps, gene_subset=sig_union
                )
            else:
                expr = load_expression_from_series_matrix_gene_symbols(
                    gse=gse,
                    cache_dir=geo_cache_dir,
                    probe_cache_dir=probe_cache_dir,
                    gene_subset=sig_union,
                )
        except Exception:
            continue

        # Align samples to endpoints
        expr = expr.loc[expr.index.intersection(sorted(sig_union))]
        keep_cols = [c for c in expr.columns if str(c) in set(sample_ids)]
        expr = expr[keep_cols]
        y2 = y.reindex([str(c) for c in expr.columns])

        if expr.shape[1] < 20 or y2.dropna().shape[0] < 20:
            continue

        z = _zscore_rows(expr)
        scores = pd.DataFrame(index=expr.columns)
        used = {}
        for e, genes in sigs.items():
            avail = [g for g in genes if g in z.index]
            used[e] = len(avail)
            if len(avail) < max(10, int(0.25 * len(genes))):
                scores[e] = np.nan
            else:
                scores[e] = z.loc[avail].mean(axis=0)

        # Record signature coverage even if this cohort cannot be analyzed downstream
        # (e.g., probe-level platforms without gene-symbol mapping in the minimal pipeline).
        for e in sigs.keys():
            coverage_rows.append(
                {
                    "dataset_id": gse,
                    "endotype": e,
                    "n_sig_genes_used": int(used.get(e, 0)),
                    "n_sig_genes_total": int(topk),
                }
            )

        scores = scores.astype(float)
        scores_z = _standardize_cols(scores)

        # Store per-sample scores for downstream LOCO and plots
        for sid, row in scores_z.iterrows():
            score_rows.append({"dataset_id": gse, "sample_id": str(sid), **{k: float(row[k]) if pd.notna(row[k]) else np.nan for k in sigs}})

        for e in sigs.keys():
            res = _glm_or_per_1sd(y2, scores_z[e])
            if res is None:
                continue
            assoc_rows.append(
                {
                    "dataset_id": gse,
                    "platform_id": platform_id,
                    "outcome": "advanced_fibrosis_F3plus_vs_F0F2",
                    "feature": e,
                    "effect_type": "OR_per_1SD",
                    "effect": res["or"],
                    "ci_lower": res["ci_lower"],
                    "ci_upper": res["ci_upper"],
                    "pvalue": res["pvalue"],
                    "log_or": res["log_or"],
                    "se": res["se"],
                    "n": res["n"],
                    "events": res["events"],
                    "n_sig_genes_used": used.get(e, 0),
                }
            )

        cohort_rows.append(
            {
                "dataset_id": gse,
                "platform_id": platform_id,
                "n_with_stage": int(y2.dropna().shape[0]),
                "n_f3plus": int(y2.dropna().sum()),
            }
        )

    assoc = pd.DataFrame(assoc_rows)
    assoc.to_csv(endotypes_dir / "endotype_fibrosis_effects_multicohort.tsv", sep="\t", index=False)
    pd.DataFrame(cohort_rows).to_csv(figures_dir / "cohort_landscape.tsv", sep="\t", index=False)
    pd.DataFrame(score_rows).to_csv(figures_dir / "endotype_scores_multicohort.tsv", sep="\t", index=False)

    # Signature coverage (cohort x endotype) used for transferability figures
    if coverage_rows:
        pd.DataFrame(coverage_rows).to_csv(figures_dir / "signature_coverage.tsv", sep="\t", index=False)

    # Spec-facing claim registry (effect sizes + uncertainty)
    if not assoc.empty:
        claim = assoc.copy()
        claim.insert(0, "claim_id", "C1")
        claim.insert(2, "model", claim["feature"].astype(str).map(lambda s: f"unadjusted:{s}"))
        claim = claim.rename(columns={"effect": "effect", "outcome": "outcome", "dataset_id": "dataset_id"})
        claim["effect_type"] = "OR_per_1SD"
        # FDR over all cohort-feature tests (conservative; explicit in docs).
        claim["fdr"] = sm.stats.multipletests(claim["pvalue"].to_numpy(dtype=float), method="fdr_bh")[1]
        claim_out = claim[
            [
                "claim_id",
                "dataset_id",
                "outcome",
                "model",
                "effect_type",
                "effect",
                "ci_lower",
                "ci_upper",
                "pvalue",
                "fdr",
                "n",
            ]
        ].copy()
        claim_out.to_csv(effect_dir / "claim_effects.tsv", sep="\t", index=False)

    # Meta-analysis
    meta_rows = []
    for e in sorted(set(assoc["feature"].astype(str))):
        d = assoc[assoc["feature"] == e].dropna(subset=["log_or", "se"])
        if d.shape[0] < 2:
            continue
        theta = d["log_or"].to_numpy(dtype=float)
        se = d["se"].to_numpy(dtype=float)
        m = dersimonian_laird(theta, se)
        meta_rows.append(
            {
                "feature": e,
                "k": int(m["k"]),
                "or_pooled": math.exp(m["mu_log_or"]),
                "ci_lower": math.exp(m["ci_lower_log"]),
                "ci_upper": math.exp(m["ci_upper_log"]),
                "prediction_interval_lower": math.exp(m["pi_lower_log"]),
                "prediction_interval_upper": math.exp(m["pi_upper_log"]),
                "I2_percent": m["I2_percent"],
                "tau2": m["tau2"],
                "Q": m["Q"],
            }
        )
    meta = pd.DataFrame(meta_rows).sort_values("feature")
    meta.to_csv(effect_dir / "meta_analysis_endotype_f3plus.tsv", sep="\t", index=False)

    # LOCO prediction (train on all other cohorts, test on held-out cohort).
    # Use the fibrogenic Endotype 1 score as the primary transportability check.
    # The three endotype scores are often strongly correlated in transferred
    # cohorts, so a multivariable score is treated as unstable for
    # cross-cohort transportability assessment.
    score_df = pd.DataFrame(score_rows)
    if not score_df.empty:
        loco_rows: list[dict[str, Any]] = []
        pred_rows: list[dict[str, Any]] = []
        coef_rows: list[dict[str, Any]] = []
        for holdout in sorted(score_df["dataset_id"].unique()):
            te = score_df[score_df["dataset_id"] == holdout].copy()
            tr = score_df[score_df["dataset_id"] != holdout].copy()

            y_te = endpoints[(endpoints["dataset_id"] == holdout)].set_index("sample_id")["y_f3plus"]
            y_tr = endpoints[(endpoints["dataset_id"] != holdout)].set_index("sample_id")["y_f3plus"]

            te = te.set_index("sample_id")
            tr = tr.set_index("sample_id")
            te = te.join(y_te.rename("y")).dropna(subset=["y"])
            tr = tr.join(y_tr.rename("y")).dropna(subset=["y"])

            features = ["endotype_1"]
            teX = te[features].to_numpy(dtype=float)
            trX = tr[features].to_numpy(dtype=float)
            tey = te["y"].to_numpy(dtype=int)
            tr_y = tr["y"].to_numpy(dtype=int)

            # Drop rows with any NaN in features
            tr_mask = np.isfinite(trX).all(axis=1)
            te_mask = np.isfinite(teX).all(axis=1)
            trX, tr_y = trX[tr_mask], tr_y[tr_mask]
            teX, tey = teX[te_mask], tey[te_mask]
            if len(tey) < 20 or len(np.unique(tey)) < 2 or len(np.unique(tr_y)) < 2:
                continue

            clf = LogisticRegression(solver="liblinear", random_state=seed)
            clf.fit(trX, tr_y)
            p = clf.predict_proba(teX)[:, 1]
            auc = float(roc_auc_score(tey, p))
            brier = float(brier_score_loss(tey, p))
            slope = float(calibration_slope(tey, p))

            # Per-sample predictions for calibration/DCA plots
            te_ids = te.index.to_numpy(dtype=str)[te_mask]
            for sid, yv, pv in zip(te_ids.tolist(), tey.tolist(), p.tolist(), strict=True):
                pred_rows.append(
                    {
                        "dataset_id": holdout,
                        "sample_id": str(sid),
                        "y_f3plus": int(yv),
                        "p": float(pv),
                        "model": "logistic_regression:endotype_1",
                    }
                )

            loco_rows.append(
                {
                    "holdout_dataset_id": holdout,
                    "auc": auc,
                    "brier": brier,
                    "calibration_slope": slope,
                    "n_test": int(len(tey)),
                    "events_test": int(tey.sum()),
                    "n_train": int(len(tr_y)),
                    "events_train": int(tr_y.sum()),
                }
            )

            # Save fold-specific coefficients for reproducibility (features are standardized within cohort).
            row: dict[str, Any] = {
                "holdout_dataset_id": holdout,
                "model": "logistic_regression:endotype_1",
                "intercept": float(clf.intercept_[0]),
                "n_train": int(len(tr_y)),
                "events_train": int(tr_y.sum()),
            }
            for name, coef in zip(features, clf.coef_[0].tolist(), strict=True):
                row[f"coef_{name}"] = float(coef)
            coef_rows.append(row)
        loco = pd.DataFrame(loco_rows)
        loco.to_csv(bench_dir / "loco_prediction_eval.tsv", sep="\t", index=False)
        pd.DataFrame(pred_rows).to_csv(bench_dir / "loco_predictions.tsv", sep="\t", index=False)
        if coef_rows:
            pd.DataFrame(coef_rows).to_csv(bench_dir / "loco_model_coefficients.tsv", sep="\t", index=False)

        # Spec-facing prediction eval table
        pred = loco.rename(columns={"holdout_dataset_id": "dataset_id", "n_test": "n"}).copy()
        pred["split_or_cohort"] = pred["dataset_id"].astype(str).map(lambda s: f"LOCO_holdout:{s}")
        pred = pred[["dataset_id", "split_or_cohort", "auc", "calibration_slope", "brier", "n"]]
        pred.to_csv(bench_dir / "prediction_eval.tsv", sep="\t", index=False)
