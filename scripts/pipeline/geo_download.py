from __future__ import annotations

from pathlib import Path

from .download import download_if_missing
from .geo_accession import parse_geo_accession


def geo_series_matrix_url(gse: str) -> str:
    g = parse_geo_accession(gse)
    if g.prefix != "GSE":
        raise ValueError(gse)
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{g.group_dir}/{g.full}/matrix/{g.full}_series_matrix.txt.gz"


def ensure_series_matrix_cached(*, gse: str, cache_dir: Path) -> Path:
    """
    Ensure the GEO series matrix is present under data/geo_cache.
    This is the minimal on-demand input needed by endpoint extraction and
    by several expression loaders (for sample/platform metadata).
    """
    dest = cache_dir / f"{gse}_series_matrix.txt.gz"
    if dest.exists():
        return dest
    url = geo_series_matrix_url(gse)
    download_if_missing(url, dest, timeout_s=120)
    return dest

