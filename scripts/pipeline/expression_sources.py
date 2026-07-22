from __future__ import annotations

import gzip
import re
import tarfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .download import download_if_missing
from .geo_download import ensure_series_matrix_cached
from .gene_id_map import GeneIdMaps, load_gene_id_maps, map_gene_id_to_symbol
from .geo_accession import parse_geo_accession
from .geo_series_matrix import read_expression_matrix, read_series_matrix_header, read_series_matrix_sample_titles
from .gpl_annotation import load_probe_to_symbol_map


_AFFY_PROBE_RE = re.compile(r"^\d+(_[a-z]+)?_at$", re.IGNORECASE)


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0, np.nan)
    return (df.sub(mu, axis=0)).div(sd, axis=0)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    return out


def _log2_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    counts = _coerce_numeric(counts).clip(lower=0)
    lib = counts.sum(axis=0).replace(0, np.nan)
    cpm = counts.div(lib, axis=1) * 1_000_000.0
    return np.log2(cpm + 1.0)


def _looks_like_gene_symbol(feature_id: str) -> bool:
    fid = (feature_id or "").strip()
    if not fid:
        return False
    # Common platform/probe identifiers must go through GPL annotation. In
    # particular, Affymetrix U133 probe sets such as 1007_s_at contain letters
    # but are not gene symbols.
    if fid.startswith(("ILMN_", "A_", "Hs.", "hsa-")):
        return False
    if _AFFY_PROBE_RE.match(fid):
        return False
    return any(c.isalpha() for c in fid)


def _geo_series_suppl_dir_url(gse: str) -> str:
    g = parse_geo_accession(gse)
    if g.prefix != "GSE":
        raise ValueError(gse)
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{g.group_dir}/{g.full}/suppl/"


def load_expression_from_series_matrix_gene_symbols(
    *,
    gse: str,
    cache_dir: Path,
    probe_cache_dir: Path,
    gene_subset: set[str] | None = None,
) -> pd.DataFrame:
    """
    Loads expression from GEO series_matrix and returns a gene_symbol x sample_id dataframe.
    If the series matrix stores probes, uses GPL annotation to map probes to gene symbols.
    """
    gz = ensure_series_matrix_cached(gse=gse, cache_dir=cache_dir)
    platform_id, sample_ids, _ = read_series_matrix_header(gz)
    feature_ids, table_sample_ids, values = read_expression_matrix(gz)
    if table_sample_ids and table_sample_ids != sample_ids:
        sample_ids = table_sample_ids

    # Heuristic: if most feature ids look like gene symbols, treat as already-gene.
    # Probe-like IDs should be mapped with the platform annotation even when they
    # contain alphabetic suffixes.
    looks_gene = sum(1 for fid in feature_ids[:200] if _looks_like_gene_symbol(str(fid)))
    as_gene = looks_gene >= max(20, int(0.6 * min(200, len(feature_ids))))

    expr = pd.DataFrame(values, index=pd.Index(feature_ids, name="feature"), columns=sample_ids)
    expr = _coerce_numeric(expr)

    if as_gene:
        expr.index = expr.index.astype(str)
        if gene_subset is not None:
            expr = expr.loc[expr.index.intersection(sorted(gene_subset))]
        return expr.groupby(level=0).mean()

    if not platform_id:
        raise RuntimeError(f"Missing platform id for {gse}; cannot map probes to genes.")
    probe_to_symbol = load_probe_to_symbol_map(gpl=platform_id.split(",")[0], cache_dir=probe_cache_dir)
    gene = [probe_to_symbol.get(fid, "") for fid in expr.index.astype(str)]
    expr = expr.assign(gene=gene)
    expr = expr[expr["gene"].astype(str) != ""].set_index("gene")
    if gene_subset is not None:
        expr = expr.loc[expr.index.intersection(sorted(gene_subset))]
    return expr.groupby(level=0).mean()


def load_expression_gse130970(
    *,
    geo_cache_dir: Path,
    suppl_cache_dir: Path,
    gene_maps: GeneIdMaps,
) -> pd.DataFrame:
    ensure_series_matrix_cached(gse="GSE130970", cache_dir=geo_cache_dir)
    url = _geo_series_suppl_dir_url("GSE130970") + "GSE130970_all_sample_salmon_tximport_TPM_entrez_gene_ID.csv.gz"
    dest = suppl_cache_dir / "GSE130970" / "tximport_TPM_entrez_gene_ID.csv.gz"
    download_if_missing(url, dest)

    df = pd.read_csv(dest, compression="gzip")
    if "entrez_id" not in df.columns:
        raise RuntimeError("Unexpected columns in GSE130970 tximport file")
    df = df.set_index("entrez_id")
    df.index = [map_gene_id_to_symbol(str(x), gene_maps) for x in df.index]
    df = df[df.index.astype(str) != ""]
    df = _coerce_numeric(df)
    df = np.log2(df.clip(lower=0) + 1.0)

    # Map sample-title columns to GSM accessions (endpoint tables use GSM).
    gz = geo_cache_dir / "GSE130970_series_matrix.txt.gz"
    _, gsm_ids, _ = read_series_matrix_header(gz)
    titles = read_series_matrix_sample_titles(gz)
    if titles and len(titles) == len(gsm_ids):
        title_to_gsm = {t: gsm for t, gsm in zip(titles, gsm_ids, strict=True)}
        cols = [title_to_gsm.get(c, c) for c in df.columns]
        df.columns = cols
        # Keep only mapped GSM columns
        keep = [c for c in df.columns if c in set(gsm_ids)]
        df = df[keep]
    return df.groupby(level=0).mean()


def load_expression_gse162694(
    *,
    geo_cache_dir: Path,
    suppl_cache_dir: Path,
    gene_maps: GeneIdMaps,
) -> pd.DataFrame:
    ensure_series_matrix_cached(gse="GSE162694", cache_dir=geo_cache_dir)
    url = _geo_series_suppl_dir_url("GSE162694") + "GSE162694_raw_counts.csv.gz"
    dest = suppl_cache_dir / "GSE162694" / "raw_counts.csv.gz"
    download_if_missing(url, dest)

    df = pd.read_csv(dest, compression="gzip", index_col=0)
    df.index = [map_gene_id_to_symbol(str(x), gene_maps) for x in df.index]
    df = df[df.index.astype(str) != ""]
    df = _coerce_numeric(df)
    df = _log2_cpm(df)

    # Columns are like 548nash1; map from the Sample_title trailing key to GSM.
    gz = geo_cache_dir / "GSE162694_series_matrix.txt.gz"
    _, gsm_ids, _ = read_series_matrix_header(gz)
    titles = read_series_matrix_sample_titles(gz)
    key_to_gsm: dict[str, str] = {}
    if titles and len(titles) == len(gsm_ids):
        for t, gsm in zip(titles, gsm_ids, strict=True):
            sample_key = (t or "").strip().split()[-1]
            if sample_key:
                key_to_gsm[sample_key] = gsm
    cols = [key_to_gsm.get(c, c) for c in df.columns]
    df.columns = cols
    keep = [c for c in df.columns if c in set(gsm_ids)]
    if keep:
        df = df[keep]
    return df.groupby(level=0).mean()


def load_expression_gse135251(
    *,
    suppl_cache_dir: Path,
    gene_maps: GeneIdMaps,
    gene_subset: set[str] | None = None,
) -> pd.DataFrame:
    url = _geo_series_suppl_dir_url("GSE135251") + "GSE135251_RAW.tar"
    dest = suppl_cache_dir / "GSE135251" / "GSE135251_RAW.tar"
    download_if_missing(url, dest, timeout_s=120)

    columns: list[str] = []
    series: list[pd.Series] = []
    with tarfile.open(dest, "r") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            if not m.name.endswith(".counts.txt.gz"):
                continue
            # Member name typically starts with GSM accession.
            sample = Path(m.name).name.split("_", 1)[0]
            f = tf.extractfile(m)
            if f is None:
                continue
            with gzip.open(f, "rt", encoding="utf-8", errors="replace") as gz:
                rows = []
                for line in gz:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    gid, val = parts[0], parts[1]
                    sym = map_gene_id_to_symbol(gid, gene_maps)
                    if not sym:
                        continue
                    if gene_subset is not None and sym not in gene_subset:
                        continue
                    try:
                        v = float(val)
                    except Exception:
                        continue
                    rows.append((sym, v))
            if not rows:
                continue
            s = pd.Series(dict(rows), name=sample, dtype=float)
            series.append(s)
            columns.append(sample)

    if not series:
        raise RuntimeError("No sample count files found inside GSE135251_RAW.tar")
    mat = pd.concat(series, axis=1).fillna(0.0)
    mat = mat.groupby(level=0).mean()
    mat = _log2_cpm(mat)
    # Column names are GSM accessions already.
    return mat


def get_default_expression_cohorts() -> list[str]:
    # Cohorts with fibrosis staging in series-matrix characteristics and a feasible expression path:
    # - series matrix (microarray): GSE48452, GSE49541, GSE89632
    # - series matrix (gene symbols): GSE163211
    # - supplementary processed matrices: GSE130970, GSE162694; raw tar counts: GSE135251
    return ["GSE163211", "GSE48452", "GSE49541", "GSE89632", "GSE130970", "GSE162694", "GSE135251"]
