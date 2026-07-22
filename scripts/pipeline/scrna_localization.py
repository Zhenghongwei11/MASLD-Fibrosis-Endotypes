from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.download import download_if_missing


@dataclass(frozen=True)
class ScrnaAtlas:
    dataset_id: str
    data_url: str
    celltypes_url: str
    data_path: Path
    celltypes_path: Path


GSE115469 = ScrnaAtlas(
    dataset_id="GSE115469",
    data_url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115469/suppl/GSE115469_Data.csv.gz",
    celltypes_url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115469/suppl/GSE115469_CellClusterType.txt.gz",
    data_path=Path("data/geo_suppl/GSE115469/GSE115469_Data.csv.gz"),
    celltypes_path=Path("data/geo_suppl/GSE115469/GSE115469_CellClusterType.txt.gz"),
)


def _ensure_inputs(repo_root: Path, atlas: ScrnaAtlas, *, allow_download: bool) -> tuple[Path, Path] | None:
    data_path = repo_root / atlas.data_path
    celltypes_path = repo_root / atlas.celltypes_path
    if data_path.exists() and celltypes_path.exists():
        return data_path, celltypes_path
    if not allow_download:
        return None
    try:
        download_if_missing(atlas.celltypes_url, celltypes_path)
        download_if_missing(atlas.data_url, data_path)
    except Exception:
        return None
    if data_path.exists() and celltypes_path.exists():
        return data_path, celltypes_path
    return None


def _load_endotype_signatures(endotype_gene_signatures_tsv: Path, *, top_n: int) -> dict[str, list[str]]:
    df = pd.read_csv(endotype_gene_signatures_tsv, sep="\t")
    df["endotype"] = df["endotype"].astype(str)
    df["gene"] = df["gene"].astype(str)
    df = df.sort_values(["endotype", "rank"], ascending=[True, True])
    sig: dict[str, list[str]] = {}
    for e, sub in df.groupby("endotype", sort=True):
        genes = sub["gene"].tolist()[:top_n]
        genes = [g for g in genes if g and g != "nan"]
        if genes:
            sig[str(e)] = genes
    return sig


def _load_celltype_map(celltypes_path: Path) -> dict[str, str]:
    df = pd.read_csv(celltypes_path, sep="\t", compression="infer")
    df = df.dropna(subset=["CellName", "CellType"])
    return {str(a): str(b) for a, b in zip(df["CellName"], df["CellType"], strict=True)}


def _pretty_celltype(s: str) -> str:
    # Keep stable identifiers, but make them human-readable for tables/plots.
    return str(s).replace("_", " ").strip()


def _read_header_cells(data_path: Path) -> list[str]:
    with gzip.open(data_path, "rt", newline="") as f:
        line = f.readline()
    header = next(csv.reader([line]))
    # First column is gene symbol (empty header).
    cells = [c.strip().strip('"') for c in header[1:]]
    return cells


def _stream_gene_rows(
    data_path: Path,
    *,
    target_genes: set[str],
    n_cells_expected: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    Returns:
      gene_to_values: gene -> vector (len n_cells_expected) float32
      genes_found: ordered list of genes found in the file
    """
    gene_to_values: dict[str, np.ndarray] = {}
    genes_found: list[str] = []
    found = 0

    with gzip.open(data_path, "rt") as f:
        _ = f.readline()  # header
        for line in f:
            if not line:
                continue
            gene, rest = line.split(",", 1)
            gene = gene.strip().strip('"')
            if gene not in target_genes:
                continue
            vals = np.fromstring(rest, sep=",", dtype=np.float32)
            if vals.size != n_cells_expected:
                # Unexpected row shape; skip defensively.
                continue
            gene_to_values[gene] = vals
            genes_found.append(gene)
            found += 1
            if found >= len(target_genes):
                # All targets captured; keep reading would be wasted work.
                break

    return gene_to_values, genes_found


def run(
    repo_root: Path,
    *,
    atlas: ScrnaAtlas = GSE115469,
    top_n: int = 50,
    allow_download: bool = True,
) -> list[Path]:
    """
    Low-compute liver scRNA localization for endotype signatures.

    Outputs:
      - results/scrna/GSE115469_gene_celltype_expression.tsv
      - results/scrna/GSE115469_endotype_celltype_localization.tsv
    """
    endotype_sig_path = repo_root / "results" / "endotypes" / "endotype_gene_signatures.tsv"
    if not endotype_sig_path.exists():
        return []

    ensured = _ensure_inputs(repo_root, atlas, allow_download=allow_download)
    if ensured is None:
        return []
    data_path, celltypes_path = ensured

    out_dir = repo_root / "results" / "scrna"
    out_dir.mkdir(parents=True, exist_ok=True)

    signatures = _load_endotype_signatures(endotype_sig_path, top_n=top_n)
    if not signatures:
        return []

    celltype_map = _load_celltype_map(celltypes_path)
    cells = _read_header_cells(data_path)
    n_cells = len(cells)

    celltypes_raw = [celltype_map.get(c) for c in cells]
    valid_mask = np.array([ct is not None and str(ct).strip() != "" for ct in celltypes_raw], dtype=bool)
    if int(valid_mask.sum()) < 100:
        return []

    celltypes_valid = [_pretty_celltype(ct) for ct, ok in zip(celltypes_raw, valid_mask, strict=True) if ok]
    celltype_levels = sorted(set(celltypes_valid))
    celltype_to_idx = {ct: i for i, ct in enumerate(celltype_levels)}
    group_idx = np.array([celltype_to_idx[ct] for ct in celltypes_valid], dtype=np.int32)
    n_groups = len(celltype_levels)
    group_ncells = np.bincount(group_idx, minlength=n_groups).astype(np.int32)

    target_genes = sorted(set(g for genes in signatures.values() for g in genes))
    gene_to_vals, genes_found = _stream_gene_rows(data_path, target_genes=set(target_genes), n_cells_expected=n_cells)
    if not gene_to_vals:
        return []

    gene_rows = []
    for gene in genes_found:
        vals = gene_to_vals[gene][valid_mask]
        sum_by = np.bincount(group_idx, weights=vals, minlength=n_groups)
        nnz_by = np.bincount(group_idx, weights=(vals > 0).astype(np.float32), minlength=n_groups)
        mean_by = sum_by / np.maximum(1, group_ncells)
        pct_by = nnz_by / np.maximum(1, group_ncells)
        for ct, nct, m, p in zip(celltype_levels, group_ncells, mean_by, pct_by, strict=True):
            gene_rows.append(
                {
                    "atlas_dataset_id": atlas.dataset_id,
                    "gene": gene,
                    "cell_type": ct,
                    "n_cells": int(nct),
                    "mean_expr": float(m),
                    "pct_expr": float(p),
                }
            )

    gene_df = pd.DataFrame(gene_rows)
    gene_out = out_dir / f"{atlas.dataset_id}_gene_celltype_expression.tsv"
    gene_df.to_csv(gene_out, sep="\t", index=False)

    # Per-gene z-score across cell types (weighted by n_cells) to highlight relative localization.
    z_df = gene_df.copy()
    z_df["w"] = z_df["n_cells"].astype(float)
    z_df["wx"] = z_df["w"] * z_df["mean_expr"].astype(float)
    z_df["wx2"] = z_df["w"] * (z_df["mean_expr"].astype(float) ** 2)
    stats = (
        z_df.groupby("gene", as_index=False)
        .agg(w=("w", "sum"), wx=("wx", "sum"), wx2=("wx2", "sum"))
        .assign(mean=lambda d: d["wx"] / d["w"])
        .assign(var=lambda d: np.maximum(1e-8, (d["wx2"] / d["w"]) - (d["mean"] ** 2)))
        .assign(sd=lambda d: np.sqrt(d["var"]))
        .loc[:, ["gene", "mean", "sd"]]
    )
    z_df = z_df.merge(stats, on="gene", how="left")
    z_df["mean_z"] = (z_df["mean_expr"].astype(float) - z_df["mean"].astype(float)) / z_df["sd"].astype(float)

    endotype_rows = []
    for endotype, genes in signatures.items():
        sub = z_df[z_df["gene"].isin(genes)].copy()
        if sub.empty:
            continue
        n_target = len(genes)
        n_found = int(sub["gene"].nunique())
        agg = (
            sub.groupby("cell_type", as_index=False)
            .agg(
                n_cells=("n_cells", "first"),
                mean_expr=("mean_expr", "mean"),
                mean_z=("mean_z", "mean"),
                mean_pct_expr=("pct_expr", "mean"),
            )
            .assign(atlas_dataset_id=atlas.dataset_id)
            .assign(endotype=endotype)
            .assign(n_genes_target=n_target)
            .assign(n_genes_found=n_found)
            .loc[
                :,
                [
                    "atlas_dataset_id",
                    "endotype",
                    "cell_type",
                    "n_cells",
                    "n_genes_target",
                    "n_genes_found",
                    "mean_expr",
                    "mean_z",
                    "mean_pct_expr",
                ],
            ]
        )
        endotype_rows.append(agg)

    if not endotype_rows:
        return [gene_out]
    endotype_df = pd.concat(endotype_rows, axis=0, ignore_index=True)

    # Stable ordering: by cell-type abundance (desc), then name.
    order = (
        endotype_df.groupby("cell_type", as_index=False)
        .agg(n=("n_cells", "max"))
        .sort_values(["n", "cell_type"], ascending=[False, True])
    )
    endotype_df["cell_type"] = pd.Categorical(endotype_df["cell_type"], categories=order["cell_type"].tolist(), ordered=True)
    endotype_df = endotype_df.sort_values(["endotype", "cell_type"])

    endotype_out = out_dir / f"{atlas.dataset_id}_endotype_celltype_localization.tsv"
    endotype_df.to_csv(endotype_out, sep="\t", index=False)

    return [gene_out, endotype_out]
