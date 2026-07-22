from __future__ import annotations

import csv
import gzip
import re
import urllib.error
from pathlib import Path

from .download import download_if_missing
from .geo_accession import parse_geo_accession


def geo_platform_annot_url(gpl: str) -> str:
    g = parse_geo_accession(gpl)
    if g.prefix != "GPL":
        raise ValueError(gpl)
    return f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{g.group_dir}/{g.full}/annot/{g.full}.annot.gz"


def geo_platform_soft_url(gpl: str) -> str:
    g = parse_geo_accession(gpl)
    if g.prefix != "GPL":
        raise ValueError(gpl)
    return f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{g.group_dir}/{g.full}/soft/{g.full}_family.soft.gz"


_RE_SPLIT_SYMBOLS = re.compile(r"\s*(///|//|;|,)\s*")


def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").strip().lower()).strip("_")


def _pick_symbol(raw: str) -> str:
    s = (raw or "").strip()
    if not s or s in {"---", "NA", "N/A", "null"}:
        return ""
    # Some platforms store multiple symbols; keep the first plausible entry.
    parts = [p.strip() for p in _RE_SPLIT_SYMBOLS.split(s) if p.strip() and p not in {"///", "//", ";", ","}]
    if not parts:
        parts = [s]
    cand = parts[0].strip()
    # Drop obvious non-symbols
    if cand in {"---", "NA", "N/A"}:
        return ""
    return cand


def _detect_columns(header: list[str]) -> tuple[int, int]:
    norm = [_norm_col(h) for h in header]
    # Probe ID column
    probe_candidates = ["id", "id_ref", "probe_id", "probeid", "ilmn_id", "il_mn_id"]
    probe_idx = -1
    for c in probe_candidates:
        if c in norm:
            probe_idx = norm.index(c)
            break
    if probe_idx < 0:
        probe_idx = 0

    # Gene symbol column
    sym_candidates = ["gene_symbol", "genesymbol", "symbol", "gene", "gene_symbol_2", "gene_symbol_1"]
    sym_idx = -1
    for c in sym_candidates:
        if c in norm:
            sym_idx = norm.index(c)
            break
    if sym_idx < 0:
        # Fallback: look for any column containing "symbol"
        for i, c in enumerate(norm):
            if "symbol" in c:
                sym_idx = i
                break
    if sym_idx < 0:
        raise RuntimeError(f"Could not detect gene symbol column in platform annotation header: {header[:20]}")

    return probe_idx, sym_idx


def _parsed_cache_path(cache_dir: Path, gpl: str) -> Path:
    return cache_dir / "gpl_annot" / f"{gpl}.probe_to_symbol.tsv"


def _read_parsed_cache(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or {"probe_id", "gene_symbol"} - set(reader.fieldnames):
            raise RuntimeError(f"Invalid parsed GPL cache header: {path}")
        for row in reader:
            probe = (row.get("probe_id") or "").strip()
            sym = (row.get("gene_symbol") or "").strip()
            if probe and sym:
                mapping[probe] = sym
    if not mapping:
        raise RuntimeError(f"Parsed GPL cache is empty: {path}")
    return mapping


def _write_parsed_cache(path: Path, mapping: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["probe_id", "gene_symbol"])
        for probe in sorted(mapping):
            writer.writerow([probe, mapping[probe]])
    tmp.replace(path)


def load_probe_to_symbol_map(
    *,
    gpl: str,
    cache_dir: Path,
) -> dict[str, str]:
    """
    Downloads and parses GEO GPL annotation into a probe_id -> gene_symbol map.
    """
    parsed_cache = _parsed_cache_path(cache_dir, gpl)
    if parsed_cache.exists():
        return _read_parsed_cache(parsed_cache)

    url = geo_platform_annot_url(gpl)
    dest = cache_dir / "gpl_annot" / f"{gpl}.annot.gz"
    try:
        download_if_missing(url, dest)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        url = geo_platform_soft_url(gpl)
        dest = cache_dir / "gpl_soft" / f"{gpl}_family.soft.gz"
        download_if_missing(url, dest, timeout_s=180)

    with gzip.open(dest, "rt", encoding="utf-8", errors="replace") as f:
        # Skip comment preamble and find the header row.
        for line in f:
            if line.startswith("!platform_table_begin"):
                line = next(f)
            if not line.startswith(("#", "^", "!", "#")) and "\t" in line:
                header = [c.strip() for c in line.rstrip("\n").split("\t")]
                break
        else:
            raise RuntimeError(f"Missing table header in {dest}")

        probe_idx, sym_idx = _detect_columns(header)
        reader = csv.reader(f, delimiter="\t")
        mapping: dict[str, str] = {}
        for row in reader:
            if not row:
                continue
            if row[0].startswith("!platform_table_end"):
                break
            if len(row) <= max(probe_idx, sym_idx):
                continue
            probe = (row[probe_idx] or "").strip()
            if not probe:
                continue
            sym = _pick_symbol(row[sym_idx])
            if not sym:
                continue
            mapping[probe] = sym
        _write_parsed_cache(parsed_cache, mapping)
        return mapping
