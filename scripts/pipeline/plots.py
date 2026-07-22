from __future__ import annotations

import os
from pathlib import Path
import textwrap

import matplotlib

# Headless-safe backend (macOS default can abort without GUI context)
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.metrics import roc_auc_score


def _ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _apply_pub_style() -> None:
    # A compact, readable publication default. Keep this lightweight and deterministic.
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "font.size": 9.0,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "lines.linewidth": 1.5,
            "grid.linewidth": 0.6,
            "grid.color": "#E6E6E6",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(ax: plt.Axes, s: str) -> None:
    ax.text(
        0.01,
        0.98,
        s,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#222222",
    )


COHORT_LABELS = {
    "GSE135251": "Govaere 2020 (GSE135251)",
    "GSE163211": "Subudhi 2022 (GSE163211)",
    "GSE162694": "Pantano 2021 (GSE162694)",
    "GSE130970": "Hoang 2019 (GSE130970)",
    "GSE49541": "Zhu 2016 (GSE49541)",
    "GSE48452": "Moylan 2014 (GSE48452)",
    "GSE89632": "Arendt 2015 (GSE89632)",
}


def _cohort_label(dataset_id: str) -> str:
    return COHORT_LABELS.get(str(dataset_id), str(dataset_id))


def plot_study_design_schematic(out_dir: Path) -> list[Path]:
    """
    Figure 1 (schematic): study design + evidence chain overview.
    This is conceptual (non-numeric), so we keep it simple and readable.
    """
    _apply_pub_style()
    _ensure(out_dir)

    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(7.6, 3.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, *, fc: str = "white") -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.9,
            edgecolor="#222222",
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.04, title, ha="left", va="top", fontsize=9.2, fontweight="bold", color="#222222")
        ax.text(x + 0.02, y + h - 0.10, body, ha="left", va="top", fontsize=8.2, color="#222222", linespacing=1.15)

    def arrow(x0: float, y0: float, x1: float, y1: float, label: str | None = None) -> None:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": "#222222", "shrinkA": 0, "shrinkB": 0},
        )
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.02, label, ha="center", va="bottom", fontsize=7.8, color="#444444")

    # Layout in figure coordinates (0..1)
    w, h = 0.225, 0.24
    y_top = 0.66
    # Keep a little extra right margin so bbox_inches='tight' doesn't clip the last box.
    x0 = 0.03
    dx = 0.235

    box(
        x0,
        y_top,
        w,
        h,
        "Cohort curation",
        "Public liver biopsies\nFibrosis stage extracted\nAdvanced fibrosis: ≥F3",
        fc="#F8FAFC",
    )
    box(
        x0 + dx,
        y_top,
        w,
        h,
        "Axis discovery",
        "Discovery cohort\nNMF rank 3\nTop-loading genes",
        fc="#F8FAFC",
    )
    box(
        x0 + 2 * dx,
        y_top,
        w,
        h,
        "Cross-cohort scoring",
        "Within-cohort z scores\nGene coverage tracked\nFixed signatures",
        fc="#F8FAFC",
    )
    box(
        x0 + 3 * dx,
        y_top,
        w,
        h,
        "Primary association",
        "Cohort-wise ORs\nRandom-effects synthesis\nPrediction intervals",
        fc="#F8FAFC",
    )

    # Arrows along top row
    for i in range(3):
        arrow(x0 + (i + 1) * dx - 0.01, y_top + h / 2, x0 + (i + 1) * dx + 0.01, y_top + h / 2)

    # Bottom row: interpretation and validation modules.
    y_bot = 0.18
    x_bench = 0.045
    x_trans = 0.285
    x_bio = 0.525
    x_sens = 0.765
    w_mod = 0.185
    h_bot = 0.28

    box(
        x_bench,
        y_bot,
        w_mod,
        h_bot,
        "Signature benchmark",
        "Published fibrosis and\nMASH/NASH signatures\nscored in same cohorts",
        fc="white",
    )
    box(
        x_trans,
        y_bot,
        w_mod,
        h_bot,
        "Transportability",
        "Held-out cohort checks\nAUC, calibration\nnet-benefit curves",
        fc="white",
    )
    box(
        x_bio,
        y_bot,
        w_mod,
        h_bot,
        "Biology annotation",
        "Pathway enrichment\nhuman liver scRNA atlas\ncell compartments",
        fc="white",
    )
    box(
        x_sens,
        y_bot,
        w_mod,
        h_bot,
        "Discovery sensitivity",
        "Alternate RNA-seq cohort\ncomponent overlap\nscore concordance",
        fc="white",
    )

    x_primary = x0 + 3 * dx
    for x_target in [x_bench, x_trans, x_bio, x_sens]:
        arrow(x_primary + w / 2, y_top, x_target + w_mod / 2, y_bot + h_bot)

    ax.text(0.02, 0.96, "Study design and evidence chain", ha="left", va="top", fontsize=10.2, fontweight="bold", color="#222222")

    pdf = out_dir / "Figure1_study_design.pdf"
    png = out_dir / "Figure1_study_design.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return [pdf, png]


def plot_sample_flow_diagram(
    dataset_endpoints_summary_tsv: Path,
    signature_coverage_tsv: Path,
    out_dir: Path,
) -> list[Path]:
    """
    Supplementary Figure S5: sample flow diagram for STROBE-style transparency.
    Uses endpoints summary (stage availability) and signature coverage to summarize exclusions.
    """
    _apply_pub_style()
    _ensure(out_dir)

    df = pd.read_csv(dataset_endpoints_summary_tsv, sep="\t")
    df["n_total"] = pd.to_numeric(df["n_total"], errors="coerce")
    df["n_with_stage"] = pd.to_numeric(df["n_with_stage"], errors="coerce")

    cov = pd.read_csv(signature_coverage_tsv, sep="\t") if signature_coverage_tsv.exists() else pd.DataFrame()
    n_total = int(np.nansum(df["n_total"].to_numpy(dtype=float)))
    n_stage = int(np.nansum(df["n_with_stage"].to_numpy(dtype=float)))
    n_missing_stage = max(0, n_total - n_stage)
    df["n_f3plus"] = pd.to_numeric(df.get("n_f3plus", 0), errors="coerce").fillna(0)
    df["n_nonadvanced"] = df["n_with_stage"].fillna(0) - df["n_f3plus"]
    no_contrast = df[(df["n_with_stage"].fillna(0) > 0) & ((df["n_f3plus"] <= 0) | (df["n_nonadvanced"] <= 0))]
    n_no_contrast = int(np.nansum(no_contrast["n_with_stage"].to_numpy(dtype=float)))

    # Exclude cohorts with essentially zero signature coverage (mean across axes).
    n_excluded_lowcov = 0
    n_not_scored = 0
    analyzable_stage = df[(df["n_f3plus"] > 0) & (df["n_nonadvanced"] > 0)].copy()
    if not cov.empty:
        cov["n_sig_genes_used"] = pd.to_numeric(cov["n_sig_genes_used"], errors="coerce")
        cov["n_sig_genes_total"] = pd.to_numeric(cov["n_sig_genes_total"], errors="coerce")
        cov["frac"] = cov["n_sig_genes_used"] / cov["n_sig_genes_total"].replace(0, np.nan)
        mean_cov = cov.groupby("dataset_id", as_index=False)["frac"].mean()
        scored = set(mean_cov["dataset_id"].astype(str))
        low = mean_cov[mean_cov["frac"].fillna(0) < 0.10]["dataset_id"].astype(str).tolist()
        if low:
            n_excluded_lowcov = int(
                np.nansum(
                    analyzable_stage[analyzable_stage["dataset_id"].astype(str).isin(low)]["n_with_stage"].to_numpy(dtype=float)
                )
            )
        not_scored = analyzable_stage[~analyzable_stage["dataset_id"].astype(str).isin(scored)]
        n_not_scored = int(np.nansum(not_scored["n_with_stage"].to_numpy(dtype=float)))
    else:
        n_not_scored = int(np.nansum(analyzable_stage["n_with_stage"].to_numpy(dtype=float)))

    n_included = max(0, n_stage - n_no_contrast - n_not_scored - n_excluded_lowcov)

    from matplotlib.patches import FancyBboxPatch

    # Wider canvas + explicit gaps between boxes to avoid label overlap.
    fig = plt.figure(figsize=(10.5, 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x: float, y: float, w: float, h: float, title: str, body: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.9,
            edgecolor="#222222",
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.05, title, ha="left", va="top", fontsize=9.2, fontweight="bold", color="#222222")
        ax.text(x + 0.02, y + h - 0.12, body, ha="left", va="top", fontsize=8.2, color="#222222", linespacing=1.15)

    def arrow(x0: float, y0: float, x1: float, y1: float, label: str | None = None) -> None:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": "#222222", "shrinkA": 0, "shrinkB": 0},
        )
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.02, label, ha="center", va="bottom", fontsize=8.0, color="#444444")

    # Main flow
    w, h = 0.25, 0.36
    gap = 0.10
    y = 0.48
    x1 = 0.03
    x2 = x1 + w + gap
    x3 = x2 + w + gap
    box(x1, y, w, h, "Selected GEO cohorts", f"Series-matrix samples\nTotal n = {n_total}")
    box(x2, y, w, h, "With explicit fibrosis stage", f"Eligible stage labels\nn = {n_stage}")
    box(x3, y, w, h, "Included in analysis", f"Adequate coverage\nn = {n_included}")

    y_mid = y + h / 2
    arrow(x1 + w, y_mid, x2, y_mid)
    arrow(x2 + w, y_mid, x3, y_mid)
    ax.text(
        x1 + w + gap / 2,
        0.27,
        f"Exclude missing stage\n(n = {n_missing_stage})",
        ha="center",
        va="center",
        fontsize=7.3,
        color="#444444",
        linespacing=1.1,
    )
    ax.text(
        x2 + w + gap / 2,
        0.25,
        "Exclude no F3+ contrast\n"
        f"(n = {n_no_contrast})\n"
        "or not scored/low coverage\n"
        f"(n = {n_not_scored + n_excluded_lowcov})",
        ha="center",
        va="center",
        fontsize=7.3,
        color="#444444",
        linespacing=1.1,
    )

    ax.text(0.02, 0.93, "Sample inclusion flow", ha="left", va="top", fontsize=10.2, fontweight="bold", color="#222222")

    pdf = out_dir / "FigureS5_flow_diagram.pdf"
    png = out_dir / "FigureS5_flow_diagram.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return [pdf, png]


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z**2) / (4 * (n**2)))
    return max(0.0, center - half), min(1.0, center + half)


def _pretty_endotype_label(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("endotype_"):
        return "Axis " + s.split("_", 1)[1]
    return s.replace("_", " ").strip()


def plot_stage_counts(stage_counts_tsv: Path, out_dir: Path) -> list[Path]:
    _apply_pub_style()
    df = pd.read_csv(stage_counts_tsv, sep="\t")
    df = df.sort_values("n", ascending=False)
    _ensure(out_dir)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(df["nafld_stage"], df["n"], color="#2F5597")
    ax.set_ylabel("Number of samples")
    ax.set_title("Cohort overview (GSE163211): NAFLD stage labels")
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    for i, v in enumerate(df["n"].tolist()):
        ax.text(i, v + max(df["n"]) * 0.01, str(int(v)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()

    pdf = out_dir / "Figure1_cohort_overview.pdf"
    png = out_dir / "Figure1_cohort_overview.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return [pdf, png]


def plot_endotype_scores_by_stage(scores_tsv: Path, out_dir: Path) -> list[Path]:
    _apply_pub_style()
    df = pd.read_csv(scores_tsv, sep="\t")
    # scores_tsv uses index_label=sample_id; pandas will name it 'sample_id' if present.
    stage_col = "nafld_stage"
    stages = [s for s in ["Normal", "Steatosis", "NASH_F0", "NASH_F1_F4"] if s in set(df[stage_col])]
    if not stages:
        stages = sorted(df[stage_col].astype(str).unique())

    _ensure(out_dir)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10.5, 3.3), sharey=False)
    for ax, col in zip(axes, ["endotype_1", "endotype_2", "endotype_3"], strict=True):
        data = [df.loc[df[stage_col] == s, col].dropna().to_numpy() for s in stages]
        ax.boxplot(data, labels=stages, showfliers=False)
        ax.set_title(col.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.set_ylabel("Standardized score (z)")
    fig.suptitle("Axis scores by NAFLD stage (GSE163211)", y=1.02, fontsize=11)
    fig.tight_layout()

    pdf = out_dir / "Figure3_endotype_scores_by_stage.pdf"
    png = out_dir / "Figure3_endotype_scores_by_stage.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return [pdf, png]


def plot_endotype_effects(effect_tsv: Path, out_dir: Path) -> list[Path]:
    _apply_pub_style()
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir.resolve()))
    df = pd.read_csv(effect_tsv, sep="\t")
    # Prefer adjusted model if present.
    df2 = df[df["model"].astype(str).str.startswith("adjusted")].copy()
    if df2.empty:
        df2 = df.copy()
    df2 = df2.sort_values("effect", ascending=False)

    y = np.arange(df2.shape[0])
    eff = df2["effect"].to_numpy()
    lo = df2["ci_lower"].to_numpy()
    hi = df2["ci_upper"].to_numpy()

    _ensure(out_dir)
    fig, ax = plt.subplots(figsize=(7.2, 2.4 + 0.35 * max(1, df2.shape[0])))
    ax.errorbar(eff, y, xerr=[eff - lo, hi - eff], fmt="o", color="black", ecolor="black", capsize=3)
    ax.axvline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df2["feature"].tolist())
    ax.set_xlabel("Odds ratio per 1-SD (95% CI)")
    ax.set_title("Axis-score association with fibrotic steatohepatitis (F1–F4 label; GSE163211)")
    ax.invert_yaxis()
    fig.tight_layout()

    pdf = out_dir / "Figure2_endotype_effects.pdf"
    png = out_dir / "Figure2_endotype_effects.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return [pdf, png]


def plot_cohort_landscape(cohort_landscape_tsv: Path, out_dir: Path) -> list[Path]:
    _apply_pub_style()
    df = pd.read_csv(cohort_landscape_tsv, sep="\t")
    df = df.sort_values("n_with_stage", ascending=False)
    _ensure(out_dir)

    total = df["n_with_stage"].to_numpy(dtype=float)
    events = df["n_f3plus"].to_numpy(dtype=float)
    rates = (events / np.maximum(total, 1.0)).astype(float)
    ci = np.array([_wilson_ci(int(k), int(n)) for k, n in zip(events, total, strict=True)], dtype=float)

    fig, (axA, axB) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(7.2, 3.15),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.28},
        sharey=True,
    )

    y = np.arange(df.shape[0])

    # Panel A: cohort sizes with events overlaid (horizontal for readability)
    axA.barh(y, total, color="#C9D5E6", label="Total (with stage)", height=0.62)
    axA.barh(y, events, color="#2F5597", label="Advanced fibrosis (≥F3)", height=0.62)
    axA.set_yticks(y)
    axA.set_yticklabels([_cohort_label(x) for x in df["dataset_id"].astype(str)], fontsize=8.4)
    axA.invert_yaxis()
    axA.set_xlabel("Number of samples")
    axA.set_title("Sample size and event counts")
    axA.legend(loc="lower right", frameon=False, fontsize=8.2)
    _panel_label(axA, "A")

    lo = ci[:, 0]
    hi = ci[:, 1]
    axB.errorbar(
        rates,
        y,
        xerr=[rates - lo, hi - rates],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=3,
    )
    axB.set_yticks(y)
    axB.tick_params(axis="y", labelleft=False)
    axB.set_xlabel("Event rate (≥F3)")
    axB.set_xlim(0.0, min(1.0, max(0.6, float(np.nanmax(hi)) + 0.05)))
    axB.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    axB.set_title("Event rate (Wilson 95% CI)")
    _panel_label(axB, "B")

    fig.tight_layout()

    pdf = out_dir / "Figure1_cohort_landscape.pdf"
    png = out_dir / "Figure1_cohort_landscape.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def plot_forest_meta(effect_tsv: Path, meta_tsv: Path, out_dir: Path) -> list[Path]:
    _apply_pub_style()
    eff = pd.read_csv(effect_tsv, sep="\t")
    meta = pd.read_csv(meta_tsv, sep="\t")
    _ensure(out_dir)

    features = [f for f in ["endotype_1", "endotype_2", "endotype_3"] if f in set(eff["feature"].astype(str))]
    if not features:
        features = sorted(set(eff["feature"].astype(str)))

    nrows = len(features)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(7.2, 1.9 + 1.55 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for idx, (ax, feat) in enumerate(zip(axes, features, strict=True)):
        d = eff[eff["feature"].astype(str) == feat].copy()
        d = d.sort_values("dataset_id")
        y = np.arange(d.shape[0] + 1)
        or_ = d["effect"].to_numpy(dtype=float)
        lo = d["ci_lower"].to_numpy(dtype=float)
        hi = d["ci_upper"].to_numpy(dtype=float)

        ax.errorbar(or_, y[:-1], xerr=[or_ - lo, hi - or_], fmt="s", color="black", ecolor="black", capsize=3)

        m = meta[meta["feature"].astype(str) == feat]
        if not m.empty:
            m = m.iloc[0]
            pooled = float(m["or_pooled"])
            plo = float(m["ci_lower"])
            phi = float(m["ci_upper"])
            # pooled as diamond
            ax.errorbar([pooled], [y[-1]], xerr=[[pooled - plo], [phi - pooled]], fmt="D", color="#2F5597", ecolor="#2F5597", capsize=4)
            ax.set_title(f"{_pretty_endotype_label(feat)}: validation OR={pooled:.2f}", fontsize=10)
        else:
            ax.set_title(f"{_pretty_endotype_label(feat)}: cohort-level ORs", fontsize=10)

        ax.axvline(1.0, linestyle="--", color="gray", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([_cohort_label(x) for x in d["dataset_id"].astype(str)] + ["Summary"])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(0.08, 5.0)
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
        _panel_label(ax, chr(ord("A") + idx))

    axes[-1].set_xlabel("Odds ratio per 1-SD transcriptomic-axis score (log scale)")
    fig.suptitle("Cohort-level validation of transcriptomic-axis associations with advanced fibrosis", y=1.01, fontsize=10.5)
    fig.tight_layout()

    pdf = out_dir / "Figure2_forest_meta.pdf"
    png = out_dir / "Figure2_forest_meta.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def _fit_calibration_params(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    p = np.clip(p.astype(float), eps, 1.0 - eps)
    x = np.log(p / (1.0 - p)).reshape(-1, 1)
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    lr.fit(x, y_true.astype(int))
    slope = float(lr.coef_.ravel()[0])
    intercept = float(lr.intercept_.ravel()[0])
    return slope, intercept


def _bootstrap_calibration_slope_ci(
    y_true: np.ndarray, p: np.ndarray, *, n_boot: int = 350, seed: int = 1
) -> tuple[float, float, float]:
    rng = default_rng(seed)
    n = int(len(y_true))
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            slope, _ = _fit_calibration_params(yb, p[idx])
        except Exception:
            continue
        if np.isfinite(slope):
            boots.append(float(slope))
    if not boots:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.quantile(np.array(boots, dtype=float), [0.025, 0.975]).tolist()
    return float(np.median(boots)), float(lo), float(hi)


def plot_loco_performance(loco_predictions_or_eval_tsv: Path, out_dir: Path) -> list[Path]:
    """
    Figure 3: external validation summary.
    Prefer per-sample predictions (loco_predictions.tsv) to show uncertainty.
    """
    _apply_pub_style()
    df = pd.read_csv(loco_predictions_or_eval_tsv, sep="\t")
    _ensure(out_dir)

    rows: list[dict[str, object]] = []
    if {"y_f3plus", "p", "dataset_id"}.issubset(set(df.columns)):
        cohorts = sorted(df["dataset_id"].astype(str).unique())
        for i, c in enumerate(cohorts):
            sub = df[df["dataset_id"].astype(str) == c].dropna(subset=["y_f3plus", "p"])
            y = sub["y_f3plus"].to_numpy(dtype=int)
            p = sub["p"].to_numpy(dtype=float)
            if len(y) < 20 or len(np.unique(y)) < 2:
                continue
            auc, lo, hi = _bootstrap_auc_ci(y, p, n_boot=500, seed=100 + i)
            slope_med, slope_lo, slope_hi = _bootstrap_calibration_slope_ci(y, p, n_boot=350, seed=200 + i)
            try:
                slope, intercept = _fit_calibration_params(y, p)
            except Exception:
                slope, intercept = float("nan"), float("nan")
            brier = float(np.mean((p - y) ** 2))
            rows.append(
                {
                    "dataset_id": c,
                    "n": int(len(y)),
                    "events": int(y.sum()),
                    "auc": float(auc),
                    "auc_lo": float(lo),
                    "auc_hi": float(hi),
                    "slope": float(slope) if np.isfinite(slope) else float("nan"),
                    "slope_lo": float(slope_lo),
                    "slope_hi": float(slope_hi),
                    "intercept": float(intercept) if np.isfinite(intercept) else float("nan"),
                    "brier": brier,
                }
            )
    else:
        if "holdout_dataset_id" in df.columns:
            df = df.rename(columns={"holdout_dataset_id": "dataset_id"})
        for _, r in df.iterrows():
            rows.append(
                {
                    "dataset_id": str(r.get("dataset_id")),
                    "n": int(r.get("n_test", r.get("n", 0)) or 0),
                    "events": int(r.get("events_test", 0) or 0),
                    "auc": float(r.get("auc", float("nan"))),
                    "auc_lo": float("nan"),
                    "auc_hi": float("nan"),
                    "slope": float(r.get("calibration_slope", float("nan"))),
                    "slope_lo": float("nan"),
                    "slope_hi": float("nan"),
                    "intercept": float("nan"),
                    "brier": float(r.get("brier", float("nan"))),
                }
            )

    met = pd.DataFrame(rows)
    if met.empty:
        return []
    met = met.sort_values("auc", ascending=False)

    nrows = int(met.shape[0])
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7.2, 1.9 + 0.44 * max(3, nrows)), sharey=True)
    y = np.arange(nrows)

    labels = [
        f"{_cohort_label(d)} (n = {n}; events = {e})"
        for d, n, e in zip(met["dataset_id"].tolist(), met["n"].tolist(), met["events"].tolist(), strict=True)
    ]

    # A) AUC forest
    ax = axes[0]
    x = met["auc"].to_numpy(dtype=float)
    lo = met["auc_lo"].to_numpy(dtype=float)
    hi = met["auc_hi"].to_numpy(dtype=float)
    if np.isfinite(lo).any() and np.isfinite(hi).any():
        ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", color="black", ecolor="black", capsize=3)
    else:
        ax.plot(x, y, "o", color="black")
    ax.axvline(0.5, linestyle="--", color="gray", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("AUC (bootstrap 95% CI)")
    ax.set_title("Discrimination")
    ax.set_xlim(0.35, 1.0)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    _panel_label(ax, "A")

    # B) Calibration slope forest
    ax = axes[1]
    s = met["slope"].to_numpy(dtype=float)
    slo = met["slope_lo"].to_numpy(dtype=float)
    shi = met["slope_hi"].to_numpy(dtype=float)
    if np.isfinite(slo).any() and np.isfinite(shi).any():
        ax.errorbar(s, y, xerr=[s - slo, shi - s], fmt="o", color="black", ecolor="black", capsize=3)
    else:
        ax.plot(s, y, "o", color="black")
    ax.axvline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Calibration slope")
    ax.set_title("Calibration")
    ax.set_xlim(0.0, 2.6)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    _panel_label(ax, "B")

    fig.suptitle("Held-out cohort transportability summary", y=1.02, fontsize=10.5)
    fig.tight_layout()

    pdf = out_dir / "Figure3_loco_prediction.pdf"
    png = out_dir / "Figure3_loco_prediction.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def _bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, *, n_boot: int = 500, seed: int = 1) -> tuple[float, float, float]:
    y = y.astype(int)
    p = p.astype(float)
    auc = float(roc_auc_score(y, p))
    rng = default_rng(seed)
    n = len(y)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(float(roc_auc_score(yb, p[idx])))
    if not boots:
        return auc, float("nan"), float("nan")
    lo, hi = np.quantile(np.array(boots, dtype=float), [0.025, 0.975]).tolist()
    return auc, float(lo), float(hi)


def _calibration_bins(y: np.ndarray, p: np.ndarray, *, n_bins: int = 10) -> pd.DataFrame:
    eps = 1e-9
    p = np.clip(p.astype(float), eps, 1 - eps)
    y = y.astype(int)
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) <= 2:
        # Fallback equal-width
        bins = np.linspace(0, 1, n_bins + 1)
    b = np.digitize(p, bins[1:-1], right=True)
    rows = []
    for k in range(int(np.max(b)) + 1):
        mask = b == k
        if mask.sum() < 10:
            continue
        rows.append(
            {
                "p_mean": float(np.mean(p[mask])),
                "y_mean": float(np.mean(y[mask])),
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def _decision_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    y = y.astype(int)
    p = p.astype(float)
    n = float(len(y))
    rows = []
    for t in thresholds.tolist():
        pred_pos = p >= t
        tp = float(np.sum((pred_pos) & (y == 1)))
        fp = float(np.sum((pred_pos) & (y == 0)))
        nb = (tp / n) - (fp / n) * (t / (1.0 - t))
        rows.append({"threshold": float(t), "net_benefit": float(nb)})
    return pd.DataFrame(rows)


def plot_prediction_srt_style(loco_predictions_tsv: Path, out_dir: Path) -> list[Path]:
    _apply_pub_style()
    """
    SRT-style multi-panel prediction figure:
      A) Prediction plot (risk distribution by outcome; pooled LOCO predictions)
      B) Calibration curves (binned reliability)
      C) Decision curve analysis (net benefit)
      D) AUC (bootstrap CI) per held-out cohort
    """
    if not loco_predictions_tsv.exists():
        return []
    df = pd.read_csv(loco_predictions_tsv, sep="\t")
    if df.empty:
        return []
    _ensure(out_dir)

    cohorts = sorted(df["dataset_id"].astype(str).unique())

    # AUC with bootstrap CI (per held-out cohort)
    auc_rows = []
    for i, c in enumerate(cohorts):
        sub = df[df["dataset_id"].astype(str) == c].dropna(subset=["y_f3plus", "p"])
        y = sub["y_f3plus"].to_numpy(dtype=int)
        p = sub["p"].to_numpy(dtype=float)
        if len(y) < 20 or len(np.unique(y)) < 2:
            continue
        auc, lo, hi = _bootstrap_auc_ci(y, p, n_boot=400, seed=10 + i)
        auc_rows.append({"dataset_id": c, "auc": auc, "lo": lo, "hi": hi, "n": int(len(y)), "events": int(y.sum())})
    auc_df = pd.DataFrame(auc_rows).sort_values("auc", ascending=False)

    pooled = df.dropna(subset=["y_f3plus", "p"])
    y_all = pooled["y_f3plus"].to_numpy(dtype=int)
    p_all = pooled["p"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.35, wspace=0.28)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[1, 0])

    # Panel A: Prediction plot (pooled predicted risks)
    p0 = p_all[y_all == 0]
    p1 = p_all[y_all == 1]
    parts = axA.violinplot([p0, p1], positions=[0, 1], showmeans=False, showmedians=True, widths=0.85)
    colors = ["#BDBDBD", "#2F5597"]
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(colors[i])
        b.set_edgecolor("#222222")
        b.set_alpha(0.85)
    for k in ["cmins", "cmaxes", "cbars", "cmedians"]:
        if k in parts:
            parts[k].set_color("#222222")
            parts[k].set_linewidth(0.8)
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["≤F2", "≥F3"], fontsize=9)
    axA.set_ylabel("Predicted risk (LOCO holdout)")
    axA.set_title("Prediction (risk distribution)")
    axA.set_ylim(0, 1)
    axA.text(
        0.07,
        0.92,
        f"Pooled held-out samples: n = {int(len(y_all))}, events = {int(y_all.sum())}",
        transform=axA.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none", "alpha": 0.75},
    )
    _panel_label(axA, "A")

    # Panel B: Calibration (reliability curves)
    ax = axB
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Ideal")
    for c in cohorts:
        sub = df[df["dataset_id"].astype(str) == c].dropna(subset=["y_f3plus", "p"])
        y = sub["y_f3plus"].to_numpy(dtype=int)
        p = sub["p"].to_numpy(dtype=float)
        if len(y) < 30:
            continue
        cal = _calibration_bins(y, p, n_bins=10)
        if cal.empty:
            continue
        ax.plot(
            cal["p_mean"],
            cal["y_mean"],
            "-",
            color="#BDBDBD",
            linewidth=1.0,
            alpha=0.55,
            zorder=1,
        )

    # Pooled calibration curve (stacked LOCO predictions)
    cal_all = _calibration_bins(y_all, p_all, n_bins=12)
    if not cal_all.empty:
        ax.plot(
            cal_all["p_mean"],
            cal_all["y_mean"],
            "o-",
            color="#2F5597",
            linewidth=2.0,
            markersize=3.8,
            label="Pooled",
            zorder=3,
        )

    # Pooled calibration summary.
    try:
        from sklearn.linear_model import LogisticRegression

        eps = 1e-6
        p_clip = np.clip(p_all, eps, 1.0 - eps)
        x = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        lr.fit(x, y_all)
        slope = float(lr.coef_.ravel()[0])
        intercept = float(lr.intercept_.ravel()[0])
        brier = float(np.mean((p_all - y_all) ** 2))
        ax.text(
            0.10,
            0.95,
            f"Pooled: slope = {slope:.2f}, intercept = {intercept:+.2f}, Brier = {brier:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none", "alpha": 0.75},
        )
    except Exception:
        pass

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration (binned)")
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=1)
    _panel_label(ax, "B")

    # Panel C: Decision curve analysis
    ax = axC
    thresholds = np.linspace(0.05, 0.8, 40)
    # Treat-none baseline
    ax.plot(thresholds, np.zeros_like(thresholds), linestyle="--", color="gray", linewidth=1, label="Treat none")

    # Thin cohort curves (context; de-emphasized)
    for c in cohorts:
        sub = df[df["dataset_id"].astype(str) == c].dropna(subset=["y_f3plus", "p"])
        y = sub["y_f3plus"].to_numpy(dtype=int)
        p = sub["p"].to_numpy(dtype=float)
        if len(y) < 30:
            continue
        dca = _decision_curve(y, p, thresholds)
        ax.plot(dca["threshold"], dca["net_benefit"], color="#BDBDBD", linewidth=1.0, alpha=0.55, zorder=1)

    # Pooled curve (primary)
    prev_all = float(np.mean(y_all))
    treat_all = prev_all - (1.0 - prev_all) * (thresholds / (1.0 - thresholds))
    ax.plot(thresholds, treat_all, linestyle=":", color="black", linewidth=1.2, label="Treat all")

    dca_all = _decision_curve(y_all, p_all, thresholds)
    ax.plot(dca_all["threshold"], dca_all["net_benefit"], color="#2F5597", linewidth=2.0, label="Pooled", zorder=3)

    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    nb_min = float(np.nanmin(dca_all["net_benefit"].to_numpy(dtype=float)))
    nb_max = float(np.nanmax(dca_all["net_benefit"].to_numpy(dtype=float)))
    ax.set_ylim(max(-0.10, nb_min - 0.04), min(0.30, nb_max + 0.06))
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=1)
    _panel_label(ax, "C")

    # Panel D: AUC forest (held-out cohorts)
    ax = axD
    y_pos = np.arange(auc_df.shape[0])
    ax.errorbar(
        auc_df["auc"].to_numpy(dtype=float),
        y_pos,
        xerr=[
            auc_df["auc"].to_numpy(dtype=float) - auc_df["lo"].to_numpy(dtype=float),
            auc_df["hi"].to_numpy(dtype=float) - auc_df["auc"].to_numpy(dtype=float),
        ],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=3,
    )
    ax.axvline(0.5, linestyle="--", color="gray", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [
            f"{_cohort_label(c)} (n = {n}; events = {e})"
            for c, n, e in zip(auc_df["dataset_id"], auc_df["n"], auc_df["events"], strict=True)
        ],
        fontsize=8.5,
    )
    ax.set_xlabel("AUC (bootstrap 95% CI)")
    ax.set_title("Discrimination (holdout)")
    ax.set_xlim(0.35, 1.0)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    _panel_label(ax, "D")

    fig.tight_layout()

    pdf = out_dir / "Figure6_prediction_beyond_roc.pdf"
    png = out_dir / "Figure6_prediction_beyond_roc.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def _scrna_compartment(cell_type: str) -> str:
    s = (cell_type or "").strip().lower()
    if s.startswith("hepatocyte"):
        return "Hepatocyte"
    if "stellate" in s:
        return "Stellate"
    if "cholangi" in s:
        return "Cholangiocyte"
    if "lsec" in s or "endothelial" in s:
        return "Endothelial"
    if "macrophage" in s:
        return "Macrophage"
    if "t cells" in s or "nk" in s:
        return "T/NK"
    if "b cells" in s or "plasma" in s:
        return "B/Plasma"
    if "erythroid" in s:
        return "Erythroid"
    return "Other"


def _scrna_short_cell_type(cell_type: str) -> str:
    s = (cell_type or "").strip()
    s = s.replace("Cells", "").strip()
    s = s.replace("alpha-beta T", "T (αβ)")
    s = s.replace("gamma-delta T", "T (γδ)")
    s = s.replace("NK-like", "NK")
    s = s.replace("Mature B", "B")
    s = s.replace("Inflammatory Macrophage", "Mac (inflam)")
    s = s.replace("Non-inflammatory Macrophage", "Mac (non)")
    s = s.replace("Portal endothelial", "Endo (portal)")
    s = s.replace("Central venous LSECs", "LSEC (central)")
    s = s.replace("Periportal LSECs", "LSEC (periportal)")
    s = s.replace("Cholangiocytes", "Cholangio")
    s = s.replace("Hepatic Stellate", "Stellate")
    if s.startswith("Hepatocyte "):
        s = "Hep " + s.split(" ", 1)[1]
    return s.strip()


def _scrna_celltype_sort_key(cell_type: str) -> tuple[int, int, str]:
    import re

    s = (cell_type or "").strip().lower()
    if s.startswith("hepatocyte"):
        m = re.search(r"(\\d+)", s)
        return (0, int(m.group(1)) if m else 0, s)
    if "cholangi" in s:
        return (1, 0, s)
    if "stellate" in s:
        return (2, 0, s)
    if "lsec" in s:
        if "periportal" in s:
            return (3, 0, s)
        if "central" in s:
            return (3, 1, s)
        return (3, 2, s)
    if "endothelial" in s:
        return (4, 0, s)
    if "macrophage" in s:
        if "non" in s:
            return (5, 1, s)
        return (5, 0, s)
    if "t cells" in s:
        if "gamma" in s:
            return (6, 1, s)
        return (6, 0, s)
    if "nk" in s:
        return (7, 0, s)
    if "b cells" in s:
        return (8, 0, s)
    if "plasma" in s:
        return (9, 0, s)
    if "erythroid" in s:
        return (10, 0, s)
    return (99, 0, s)


def plot_scrna_localization(endotype_celltype_tsv: Path, out_dir: Path) -> list[Path]:
    """
    scRNA localization summary (two-resolution view):
      - Panel A: compartment-level heatmap (compact)
      - Panel B: cell-type dot plot (resolution)
        color = mean z-scored expression (across cell types)
        size  = mean fraction of genes expressed
    """
    if not endotype_celltype_tsv.exists():
        return []
    df = pd.read_csv(endotype_celltype_tsv, sep="\t")
    if df.empty:
        return []
    _apply_pub_style()
    _ensure(out_dir)

    df["endotype"] = df["endotype"].astype(str)
    df["cell_type"] = df["cell_type"].astype(str)
    df["mean_z"] = df["mean_z"].to_numpy(dtype=float)
    df["mean_pct_expr"] = df["mean_pct_expr"].to_numpy(dtype=float)

    endotypes = sorted(df["endotype"].unique())
    cell_types = sorted(pd.unique(df["cell_type"]).tolist(), key=_scrna_celltype_sort_key)

    df["compartment"] = df["cell_type"].map(_scrna_compartment)
    comp_order = ["Hepatocyte", "Stellate", "Cholangiocyte", "Endothelial", "Macrophage", "T/NK", "B/Plasma", "Erythroid", "Other"]
    comp_df = (
        df.groupby(["endotype", "compartment"], as_index=False)
        .agg(mean_z=("mean_z", "mean"), mean_pct_expr=("mean_pct_expr", "mean"))
        .assign(compartment=lambda d: pd.Categorical(d["compartment"], categories=comp_order, ordered=True))
        .sort_values(["endotype", "compartment"])
    )

    x_map = {ct: i for i, ct in enumerate(cell_types)}
    y_map = {e: i for i, e in enumerate(endotypes)}
    df["x"] = df["cell_type"].map(x_map).astype(int)
    df["y"] = df["endotype"].map(y_map).astype(int)

    # Scale dot sizes by percent-expressing (0..1) into points^2.
    size = 20.0 + 240.0 * np.clip(df["mean_pct_expr"].to_numpy(dtype=float), 0.0, 1.0)
    z = np.clip(df["mean_z"].to_numpy(dtype=float), -2.5, 2.5)

    fig_w = min(9.0, max(7.2, 0.29 * len(cell_types)))
    fig, (axA, axB) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(fig_w, 3.35 + 0.35 * max(1, len(endotypes))),
        gridspec_kw={"width_ratios": [0.95, 1.85], "wspace": 0.16},
    )

    # Panel A: compartment-level heatmap
    comp_levels = [c for c in comp_order if c in set(comp_df["compartment"].astype(str))]
    endo_labels = [_pretty_endotype_label(e) for e in endotypes]
    mat = np.full((len(endotypes), len(comp_levels)), fill_value=np.nan, dtype=float)
    for i, e in enumerate(endotypes):
        sub = comp_df[comp_df["endotype"].astype(str) == str(e)]
        for j, c in enumerate(comp_levels):
            row = sub[sub["compartment"].astype(str) == str(c)]
            if not row.empty:
                mat[i, j] = float(row["mean_z"].iloc[0])
    axA.imshow(np.clip(mat, -2.5, 2.5), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    axA.set_xticks(np.arange(len(comp_levels)))
    axA.set_xticklabels(comp_levels, rotation=25, ha="right", fontsize=7.6)
    axA.set_yticks(np.arange(len(endotypes)))
    axA.set_yticklabels(endo_labels, fontsize=8.2)
    axA.set_xlabel("Cell compartment")
    axA.set_ylabel("Axis")
    axA.set_title("Compartment-level")
    _panel_label(axA, "A")

    # Panel B: cell-type dot plot
    sc = axB.scatter(
        df["x"].to_numpy(dtype=float),
        df["y"].to_numpy(dtype=float),
        c=z,
        s=size,
        cmap="RdBu_r",
        vmin=-2.5,
        vmax=2.5,
        edgecolors="#222222",
        linewidths=0.2,
    )
    axB.set_xticks(range(len(cell_types)))
    axB.set_xticklabels([_scrna_short_cell_type(ct) for ct in cell_types], rotation=45, ha="right", fontsize=7.2)
    axB.set_yticks(np.arange(len(endotypes)))
    axB.set_yticklabels(endo_labels, fontsize=8.2)
    axB.tick_params(axis="y", labelleft=False)
    axB.set_xlabel("Liver cell type (scRNA atlas)")
    axB.set_title("Cell-type resolution")
    _panel_label(axB, "B")

    cbar = fig.colorbar(sc, ax=[axA, axB], fraction=0.035, pad=0.02)
    cbar.set_label("Mean expression (z-score; across cell types)")

    # Dot-size legend (3 anchors), outside the plotting region.
    handles = []
    labels = []
    for p in [0.2, 0.5, 0.8]:
        h = axB.scatter([], [], s=20.0 + 240.0 * p, c="#CCCCCC", edgecolors="#222222", linewidths=0.2)
        handles.append(h)
        labels.append(f"{int(p*100)}%")
    axB.legend(
        handles,
        labels,
        title="Mean % genes expressed",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        frameon=False,
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    fig.tight_layout(pad=0.6)
    pdf = out_dir / "Figure7_scrna_localization.pdf"
    png = out_dir / "Figure7_scrna_localization.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def plot_signature_coverage(signature_coverage_tsv: Path, out_dir: Path) -> list[Path]:
    if not signature_coverage_tsv.exists():
        return []
    df = pd.read_csv(signature_coverage_tsv, sep="\t")
    if df.empty:
        return []
    _ensure(out_dir)

    df["endotype"] = df["endotype"].astype(str)
    df["dataset_id"] = df["dataset_id"].astype(str)
    df["frac"] = df["n_sig_genes_used"].to_numpy(dtype=float) / df["n_sig_genes_total"].to_numpy(dtype=float)

    cohorts = sorted(df["dataset_id"].unique())
    endotypes = sorted(df["endotype"].unique())

    mat = np.full((len(cohorts), len(endotypes)), fill_value=np.nan, dtype=float)
    used_mat = np.full((len(cohorts), len(endotypes)), fill_value=np.nan, dtype=float)
    for i, c in enumerate(cohorts):
        for j, e in enumerate(endotypes):
            sub = df[(df["dataset_id"] == c) & (df["endotype"] == e)]
            if not sub.empty:
                mat[i, j] = float(sub["frac"].iloc[0])
                used_mat[i, j] = float(sub["n_sig_genes_used"].iloc[0])

    _apply_pub_style()
    fig_h = max(3.4, 0.46 * len(cohorts))
    fig, (axA, axB) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(7.6, fig_h),
        gridspec_kw={"width_ratios": [1.35, 0.9], "wspace": 0.36},
    )

    xs, ys, vals, sizes = [], [], [], []
    for i in range(len(cohorts)):
        for j in range(len(endotypes)):
            if np.isfinite(mat[i, j]):
                xs.append(j)
                ys.append(i)
                vals.append(mat[i, j])
                sizes.append(180 + 360 * mat[i, j])
    sc = axA.scatter(
        xs,
        ys,
        c=vals,
        s=sizes,
        cmap="viridis",
        vmin=0.92,
        vmax=1.0,
        edgecolors="#222222",
        linewidths=0.45,
    )
    axA.set_xlim(-0.55, len(endotypes) - 0.45)
    axA.set_ylim(len(cohorts) - 0.55, -0.55)
    axA.set_xticks(np.arange(len(endotypes)))
    axA.set_xticklabels([_pretty_endotype_label(e) for e in endotypes], rotation=0, ha="center", fontsize=8.4)
    axA.set_yticks(np.arange(len(cohorts)))
    axA.set_yticklabels([_cohort_label(c) for c in cohorts], fontsize=8.4)
    axA.set_title("Signature coverage by cohort", fontsize=9.2)
    axA.set_xlabel("Transferred axis signature")
    axA.set_facecolor("#FAFAFA")
    axA.grid(which="major", color="#E9E9E9", linewidth=0.7)
    for i in range(len(cohorts)):
        for j in range(len(endotypes)):
            if np.isfinite(mat[i, j]):
                pct = int(round(100 * mat[i, j]))
                used = int(round(used_mat[i, j])) if np.isfinite(used_mat[i, j]) else 0
                text_color = "white" if mat[i, j] < 0.975 else "#111111"
                axA.text(j, i, f"{pct}%\n{used}/60", ha="center", va="center", fontsize=6.4, color=text_color)
    cbar = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.02)
    cbar.set_label("Coverage", fontsize=8.2)
    _panel_label(axA, "A")

    mean_cov = np.nanmean(mat, axis=1)
    y = np.arange(len(cohorts))
    axB.scatter(mean_cov, y, s=34, color="#0072B2", edgecolors="#222222", linewidths=0.3, zorder=3)
    for yy, v in zip(y, mean_cov, strict=True):
        axB.plot([0.92, v], [yy, yy], color="#9AA4AA", linewidth=1.0, zorder=2)
        axB.text(min(1.002, v + 0.004), yy, f"{v*100:.1f}%", va="center", ha="left", fontsize=7.0, color="#222222")
    axB.set_yticks(y)
    axB.set_yticklabels([])
    axB.tick_params(axis="y", left=False, labelleft=False)
    axB.invert_yaxis()
    axB.spines["left"].set_visible(False)
    axB.set_xlabel("Mean coverage")
    axB.set_xlim(0.92, 1.03)
    axB.set_xticks([0.925, 0.95, 0.975, 1.0])
    axB.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    axB.set_title("Mean across axes", fontsize=9.2)
    _panel_label(axB, "B")

    fig.suptitle("Transfer scoring coverage", y=1.01, fontsize=9.8)
    fig.tight_layout()

    pdf = out_dir / "Figure4_signature_gene_coverage.pdf"
    png = out_dir / "Figure4_signature_gene_coverage.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def plot_signature_benchmarking(
    endotype_meta_tsv: Path,
    prior_meta_tsv: Path,
    correlations_tsv: Path,
    out_dir: Path,
) -> list[Path]:
    if not endotype_meta_tsv.exists() or not prior_meta_tsv.exists() or not correlations_tsv.exists():
        return []
    endo = pd.read_csv(endotype_meta_tsv, sep="\t")
    prior = pd.read_csv(prior_meta_tsv, sep="\t")
    corr = pd.read_csv(correlations_tsv, sep="\t")
    if endo.empty or prior.empty:
        return []
    _ensure(out_dir)
    _apply_pub_style()

    selected = [
        "Axis 1",
        "Fibrosis signature",
        "NASH signature",
        "HSC TGF-beta signature",
        "Macrophages C2",
        "Profibrotic Macrophages",
        "Hepatocytes",
    ]
    e1 = endo[endo["feature"].astype(str) == "endotype_1"].copy()
    rows = []
    if not e1.empty:
        r = e1.iloc[0]
        rows.append(
            {
                "label": "Axis 1",
                "or_pooled": float(r["or_pooled"]),
                "ci_lower": float(r["ci_lower"]),
                "ci_upper": float(r["ci_upper"]),
                "k": int(r["k"]),
                "median_auc": np.nan,
            }
        )
    for sig in selected[1:]:
        d = prior[prior["signature"].astype(str) == sig]
        if d.empty:
            continue
        r = d.iloc[0]
        rows.append(
            {
                "label": sig,
                "or_pooled": float(r["or_pooled"]),
                "ci_lower": float(r["ci_lower"]),
                "ci_upper": float(r["ci_upper"]),
                "k": int(r["k"]),
                "median_auc": float(r["median_auc"]),
            }
        )
    plot_df = pd.DataFrame(rows)
    plot_df["label"] = pd.Categorical(plot_df["label"], categories=selected, ordered=True)
    plot_df = plot_df.sort_values("label")

    fig, (axA, axB) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8.2, 4.0),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.35},
    )

    y = np.arange(plot_df.shape[0])
    x = plot_df["or_pooled"].to_numpy(float)
    lo = plot_df["ci_lower"].to_numpy(float)
    hi = plot_df["ci_upper"].to_numpy(float)
    colors = ["#2F5597" if lab == "Axis 1" else "#777777" for lab in plot_df["label"].astype(str)]
    axA.errorbar(x, y, xerr=[x - lo, hi - x], fmt="none", ecolor="#444444", elinewidth=1.0, capsize=2.5)
    axA.scatter(x, y, s=38, c=colors, edgecolors="#222222", linewidths=0.3, zorder=3)
    axA.axvline(1.0, linestyle="--", color="#999999", linewidth=1.0)
    axA.set_xscale("log")
    axA.set_yticks(y)
    axA.set_yticklabels(plot_df["label"].astype(str), fontsize=8.2)
    axA.invert_yaxis()
    axA.set_xlabel("Summary OR for advanced fibrosis")
    axA.set_title("Published signature benchmark", fontsize=9.5)
    axA.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.65)
    _panel_label(axA, "A")

    corr_sel = corr[corr["signature"].isin(selected[1:])].copy()
    if not corr_sel.empty:
        summary = (
            corr_sel.groupby("signature", as_index=False)
            .agg(median_r=("pearson_r", "median"), q1=("pearson_r", lambda s: s.quantile(0.25)), q3=("pearson_r", lambda s: s.quantile(0.75)), n=("pearson_r", "count"))
        )
        summary["signature"] = pd.Categorical(summary["signature"], categories=selected[1:], ordered=True)
        summary = summary.sort_values("signature")
        yy = np.arange(summary.shape[0])
        med = summary["median_r"].to_numpy(float)
        q1 = summary["q1"].to_numpy(float)
        q3 = summary["q3"].to_numpy(float)
        axB.errorbar(med, yy, xerr=[med - q1, q3 - med], fmt="o", color="#2F5597", ecolor="#777777", capsize=2.5, markersize=4.2)
        axB.axvline(0, linestyle="--", color="#999999", linewidth=1.0)
        axB.set_yticks(yy)
        axB.set_yticklabels(summary["signature"].astype(str), fontsize=8.2)
        axB.invert_yaxis()
        axB.set_xlim(-1, 1)
        axB.set_xlabel("Correlation with Axis 1")
        axB.set_title("Score concordance with Axis 1", fontsize=9.5)
        axB.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.65)
    _panel_label(axB, "B")

    fig.tight_layout()
    pdf = out_dir / "Figure5_signature_benchmarking.pdf"
    png = out_dir / "Figure5_signature_benchmarking.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def plot_pathway_annotation(enrichment_tsv: Path, out_dir: Path) -> list[Path]:
    if not enrichment_tsv.exists():
        return []
    df = pd.read_csv(enrichment_tsv, sep="\t")
    if df.empty:
        return []
    _ensure(out_dir)
    _apply_pub_style()

    # Keep the strongest non-duplicated terms per axis for a compact supplement.
    def clean_term(s: str) -> str:
        s = str(s).split(" R-HSA-")[0]
        s = s.replace("MSigDB Hallmark 2020", "").strip()
        s = s.replace("Response To Elevated Platelet Cytosolic Ca2+", "Platelet Ca2+ response")
        s = s.replace("Platelet Activation, Signaling And Aggregation", "Platelet activation and aggregation")
        s = s.replace("Formation Of Fibrin Clot (Clotting Cascade)", "Fibrin clot formation")
        s = s.replace("Post-translational Protein Phosphorylation", "Protein phosphorylation")
        s = s.replace("Binding And Uptake Of Ligands By Scavenger Receptors", "Scavenger receptor ligand uptake")
        s = s.replace("Regulation Of IGF Transport And Uptake By IGFBPs", "IGF transport by IGFBPs")
        return textwrap.fill(s[:58], width=26)

    rows = []
    term_order: list[str] = []
    for e, d in df.sort_values(["fdr", "pvalue"]).groupby("endotype"):
        seen: set[str] = set()
        for _, r in d.iterrows():
            term = clean_term(r["gene_set"])
            key = term.lower().replace(" ", "")
            if key in seen:
                continue
            seen.add(key)
            rows.append({**r.to_dict(), "term": term})
            if term not in term_order:
                term_order.append(term)
            if len(seen) >= 6:
                break
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return []

    endotypes = sorted(plot_df["endotype"].astype(str).unique())
    term_order = sorted(term_order, key=lambda t: (plot_df.loc[plot_df["term"] == t, "fdr"].min(), t), reverse=True)
    term_to_y = {term: i for i, term in enumerate(term_order)}
    endo_to_x = {e: i for i, e in enumerate(endotypes)}
    plot_df["x"] = plot_df["endotype"].map(endo_to_x)
    plot_df["y"] = plot_df["term"].map(term_to_y)
    plot_df["score"] = -np.log10(plot_df["fdr"].clip(lower=1e-300))

    fig_h = max(4.0, 0.25 * len(term_order) + 1.2)
    fig, ax = plt.subplots(figsize=(7.6, fig_h))
    size = 30 + 28 * plot_df["overlap_size"].to_numpy(float)
    sc = ax.scatter(
        plot_df["x"],
        plot_df["y"],
        s=size,
        c=plot_df["score"],
        cmap="cividis",
        edgecolors="#222222",
        linewidths=0.35,
        vmin=max(0, float(plot_df["score"].min()) - 0.25),
        vmax=float(plot_df["score"].max()),
    )
    ax.set_xticks(range(len(endotypes)))
    ax.set_xticklabels([_pretty_endotype_label(e) for e in endotypes], fontsize=8.6)
    ax.set_yticks(range(len(term_order)))
    ax.set_yticklabels(term_order, fontsize=6.8)
    ax.set_xlim(-0.55, len(endotypes) - 0.45)
    ax.set_ylim(-0.7, len(term_order) - 0.3)
    ax.set_xlabel("Axis signature")
    ax.set_title("Pathway enrichment profile", fontsize=9.5)
    ax.grid(color="#E7E7E7", linewidth=0.65)
    ax.set_facecolor("#FAFAFA")
    _panel_label(ax, "A")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("-log10(FDR)")
    handles = []
    labels = []
    for ov in [4, 8, 14]:
        handles.append(ax.scatter([], [], s=30 + 28 * ov, color="#C8C8C8", edgecolors="#222222", linewidths=0.35))
        labels.append(str(ov))
    ax.legend(handles, labels, title="Overlap genes", frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout()

    pdf = out_dir / "FigureS2_pathway_annotation.pdf"
    png = out_dir / "FigureS2_pathway_annotation.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def plot_robustness_sensitivity(nmf_sensitivity_tsv: Path, out_dir: Path) -> list[Path]:
    if not nmf_sensitivity_tsv.exists():
        return []
    df = pd.read_csv(nmf_sensitivity_tsv, sep="\t")
    if df.empty:
        return []
    _ensure(out_dir)

    # Summarize by (k, component); NNDSVDa initialization is deterministic.
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df.dropna(subset=["k", "effect", "ci_lower", "ci_upper"])
    ks = sorted(int(x) for x in df["k"].unique())

    _apply_pub_style()
    fig, axes = plt.subplots(nrows=1, ncols=len(ks), figsize=(7.2, 3.0), sharey=True)
    if len(ks) == 1:
        axes = [axes]

    for idx, (ax, k) in enumerate(zip(axes, ks, strict=True)):
        d = df[df["k"] == k].copy()
        comps = sorted(d["component"].astype(str).unique())
        ys = np.arange(len(comps))
        for y, comp in zip(ys, comps, strict=True):
            s = d[d["component"].astype(str) == comp].copy()
            eff = s["effect"].to_numpy(dtype=float)
            lo = s["ci_lower"].to_numpy(dtype=float)
            hi = s["ci_upper"].to_numpy(dtype=float)
            ax.errorbar(
                eff,
                np.full_like(eff, y, dtype=float),
                xerr=[eff - lo, hi - eff],
                fmt="o",
                color="black",
                alpha=0.55,
                markersize=3.5,
            )
        ax.axvline(1.0, linestyle="--", color="gray", linewidth=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([_pretty_endotype_label(c) for c in comps], fontsize=9)
        ax.set_xscale("log")
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_title(f"Rank {k}", fontsize=10)
        _panel_label(ax, chr(ord("A") + idx))

    axes[0].set_xlabel("Odds ratio per 1-SD (log scale)")
    fig.suptitle("Discovery-cohort NMF rank sensitivity", y=1.02, fontsize=10.5)
    fig.tight_layout()

    pdf = out_dir / "Figure5_robustness_nmf_sensitivity.pdf"
    png = out_dir / "Figure5_robustness_nmf_sensitivity.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]
