from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import NMF
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .geo_series_matrix import read_series_matrix_header, read_expression_matrix


GSE = "GSE163211"


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _coerce_float(x: str) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _coerce_int(x: str) -> int | None:
    v = _coerce_float(x)
    return None if v is None or math.isnan(v) else int(round(v))


def _coerce_bool(x: str) -> int | None:
    s = (x or "").strip().lower()
    if not s:
        return None
    if s in {"yes", "y", "true", "1", "diabetes", "t2d", "dm"}:
        return 1
    if s in {"no", "n", "false", "0", "none"}:
        return 0
    return None


def load_gse163211_from_cache(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    gz = cache_dir / f"{GSE}_series_matrix.txt.gz"
    if not gz.exists():
        raise FileNotFoundError(gz)

    platform_id, sample_ids, characteristics = read_series_matrix_header(gz)

    # Expression: gene_symbol x sample
    feature_ids, table_sample_ids, values = read_expression_matrix(gz)
    if table_sample_ids != sample_ids:
        # Fallback: prefer expression header sample list
        sample_ids = table_sample_ids

    expr = pd.DataFrame(values, index=pd.Index(feature_ids, name="gene"), columns=sample_ids)
    # Deduplicate gene symbols (mean)
    expr = expr.groupby(level=0).mean()

    meta = pd.DataFrame({"sample_id": sample_ids}).set_index("sample_id")
    for key, vals in characteristics.items():
        if len(vals) != len(sample_ids):
            continue
        meta[key] = vals

    return expr, meta, platform_id


def build_endpoints(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    stage = out.get("nafld stage")
    if stage is None:
        raise RuntimeError("Missing 'nafld stage' in GSE163211 sample characteristics.")

    stage = stage.astype(str)
    out["nafld_stage"] = stage

    # Primary binary endpoint for this dataset: fibrotic NASH (F1-F4) vs others.
    out["y_fibrotic_nash"] = (stage == "NASH_F1_F4").astype(int)

    # Secondary: NASH (any) vs non-NASH
    out["y_nash_any"] = stage.isin(["NASH_F0", "NASH_F1_F4"]).astype(int)

    return out


def select_variable_genes(expr: pd.DataFrame, n: int = 5000) -> pd.DataFrame:
    # log1p stabilizes heavy-tailed counts
    x = np.log1p(expr)
    var = x.var(axis=1)
    keep = var.sort_values(ascending=False).head(n).index
    return x.loc[keep]


def fit_nmf(expr_log: pd.DataFrame, k: int = 3, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    # samples x genes
    X = expr_log.T.to_numpy(dtype=float)
    # NMF expects non-negative; log1p is non-negative.
    model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=2000)
    W = model.fit_transform(X)  # samples x k
    H = model.components_  # k x genes

    W_df = pd.DataFrame(W, index=expr_log.columns, columns=[f"endotype_{i+1}" for i in range(k)])
    H_df = pd.DataFrame(H, index=[f"endotype_{i+1}" for i in range(k)], columns=expr_log.index)
    return W_df, H_df


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean(axis=0)) / df.std(axis=0, ddof=0).replace(0, np.nan)


def run_logistic_association(
    *,
    y: pd.Series,
    X: pd.DataFrame,
    covariates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        design = pd.DataFrame({col: X[col]})
        if covariates is not None:
            for c in covariates.columns:
                design[c] = covariates[c]
        design = sm.add_constant(design, has_constant="add")
        data = pd.concat([y, design], axis=1).dropna()
        y2 = data[y.name]
        X2 = data.drop(columns=[y.name])

        model = sm.GLM(y2, X2, family=sm.families.Binomial())
        fit = model.fit()
        coef = float(fit.params[col])
        se = float(fit.bse[col])
        or_ = math.exp(coef)
        ci_lo = math.exp(coef - 1.96 * se)
        ci_hi = math.exp(coef + 1.96 * se)
        p = float(fit.pvalues[col])
        rows.append(
            {
                "feature": col,
                "model": "adjusted" if covariates is not None and covariates.shape[1] else "unadjusted",
                "effect_type": "OR_per_1SD",
                "effect": or_,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "pvalue": p,
                "n": int(data.shape[0]),
                "events": int(y2.sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["fdr"] = sm.stats.multipletests(out["pvalue"].to_numpy(), method="fdr_bh")[1]
    return out.sort_values(["model", "feature"]).reset_index(drop=True)


def cross_validated_prediction(
    *,
    y: pd.Series,
    X: pd.DataFrame,
    seed: int = 1,
    n_splits: int = 5,
) -> dict[str, Any]:
    data = pd.concat([y, X], axis=1).dropna()
    yv = data[y.name].to_numpy(dtype=int)
    Xv = data.drop(columns=[y.name]).to_numpy(dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probs = np.full_like(yv, fill_value=np.nan, dtype=float)
    for tr, te in skf.split(Xv, yv):
        Xtr = sm.add_constant(Xv[tr], has_constant="add")
        Xte = sm.add_constant(Xv[te], has_constant="add")
        fit = sm.GLM(yv[tr], Xtr, family=sm.families.Binomial()).fit()
        probs[te] = fit.predict(Xte)

    auc = roc_auc_score(yv, probs)
    brier = brier_score_loss(yv, probs)

    # Calibration slope: logistic regression of y on logit(p)
    eps = 1e-6
    logit = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))
    cal_fit = sm.GLM(yv, sm.add_constant(logit, has_constant="add"), family=sm.families.Binomial()).fit()
    cal_slope = float(cal_fit.params[1])

    return {"auc": float(auc), "brier": float(brier), "calibration_slope": cal_slope, "n": int(len(yv))}


def run(out_root: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(out_root).resolve()))

    cache_dir = Path("data/geo_cache")
    expr, meta, platform_id = load_gse163211_from_cache(cache_dir)
    meta2 = build_endpoints(meta)

    # Covariates (best-effort)
    cov = pd.DataFrame(index=meta2.index)
    cov["age"] = meta2.get("age").map(_coerce_float) if "age" in meta2.columns else np.nan
    cov["bmi"] = meta2.get("bmi").map(_coerce_float) if "bmi" in meta2.columns else np.nan
    cov["diabetes"] = meta2.get("diabetes").map(_coerce_bool) if "diabetes" in meta2.columns else np.nan
    cov["sex_male"] = (meta2.get("Sex").astype(str).str.lower() == "male").astype(float) if "Sex" in meta2.columns else np.nan

    expr_log = select_variable_genes(expr, n=5000)
    W, H = fit_nmf(expr_log, k=3, seed=1)
    Wz = _standardize(W)

    results_dir = out_root / "results"
    endotypes_dir = results_dir / "endotypes"
    effect_dir = results_dir / "effect_sizes"
    bench_dir = results_dir / "benchmarks"
    figures_anchor_dir = results_dir / "figures"
    _ensure_dirs(endotypes_dir, effect_dir, bench_dir, figures_anchor_dir)

    # Endotype definitions (top genes)
    topk = 60
    defs = []
    for e in H.index:
        weights = H.loc[e].sort_values(ascending=False)
        for rank, (gene, w) in enumerate(weights.head(topk).items(), start=1):
            defs.append({"endotype": e, "rank": rank, "gene": gene, "weight": float(w)})
    pd.DataFrame(defs).to_csv(endotypes_dir / "endotype_gene_weights.tsv", sep="\t", index=False)

    # Primary endpoint association
    y = meta2["y_fibrotic_nash"].rename("y_fibrotic_nash")
    assoc_unadj = run_logistic_association(y=y, X=Wz, covariates=None)
    assoc_adj = run_logistic_association(y=y, X=Wz, covariates=cov)
    assoc = pd.concat([assoc_unadj, assoc_adj], axis=0, ignore_index=True)

    assoc_out = assoc.copy()
    assoc_out.insert(0, "dataset_id", GSE)
    assoc_out.insert(1, "outcome", "fibrotic_NASH_F1_F4_vs_others")
    assoc_out.to_csv(endotypes_dir / "endotype_response_effects.tsv", sep="\t", index=False)

    # Claim effect table (align to spec)
    claim_rows = []
    for _, r in assoc_out.iterrows():
        claim_rows.append(
            {
                "claim_id": "C1",
                "dataset_id": r["dataset_id"],
                "outcome": r["outcome"],
                "model": r["model"] + ":" + r["feature"],
                "effect_type": r["effect_type"],
                "effect": r["effect"],
                "ci_lower": r["ci_lower"],
                "ci_upper": r["ci_upper"],
                "pvalue": r["pvalue"],
                "fdr": r["fdr"],
                "n": r["n"],
            }
        )
    pd.DataFrame(claim_rows).to_csv(effect_dir / "claim_effects.tsv", sep="\t", index=False)

    # Prediction evaluation (CV)
    pred = cross_validated_prediction(y=y, X=Wz, seed=1, n_splits=5)
    pd.DataFrame(
        [
            {
                "dataset_id": GSE,
                "split_or_cohort": "5fold_cv",
                "auc": pred["auc"],
                "calibration_slope": pred["calibration_slope"],
                "brier": pred["brier"],
                "n": pred["n"],
            }
        ]
    ).to_csv(bench_dir / "prediction_eval.tsv", sep="\t", index=False)

    # Anchor tables for figures
    stage_counts = (
        meta2[["nafld_stage"]]
        .assign(n=1)
        .groupby("nafld_stage", as_index=False)["n"]
        .sum()
        .sort_values("n", ascending=False)
    )
    stage_counts.to_csv(figures_anchor_dir / "cohort_stage_counts.tsv", sep="\t", index=False)

    plot_frame = pd.concat([meta2[["nafld_stage", "y_fibrotic_nash"]], Wz], axis=1)
    plot_frame.to_csv(figures_anchor_dir / "endotype_scores.tsv", sep="\t", index_label="sample_id")

    # Dataset summary row (this dataset only; global builder will append others)
    summary = {
        "dataset_id": GSE,
        "platform_id": platform_id,
        "n_total": int(meta2.shape[0]),
        "n_fibrotic_nash": int(meta2["y_fibrotic_nash"].sum()),
        "stage_levels": "|".join(sorted(meta2["nafld_stage"].astype(str).unique())),
        "endpoint_primary": "fibrotic_NASH_F1_F4_vs_others",
    }
    (results_dir / "dataset_summary_gse163211.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
