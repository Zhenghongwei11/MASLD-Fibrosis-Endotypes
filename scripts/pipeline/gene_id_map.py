from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

from .download import download_if_missing


NCBI_HS_GENE_INFO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"


@dataclass(frozen=True)
class GeneIdMaps:
    entrez_to_symbol: dict[str, str]
    ensembl_to_symbol: dict[str, str]


def _strip_ensembl_version(ensg: str) -> str:
    s = (ensg or "").strip()
    if "." in s:
        return s.split(".", 1)[0]
    return s


def load_gene_id_maps(cache_dir: Path) -> GeneIdMaps:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "ncbi_gene_info" / "Homo_sapiens.gene_info.gz"
    download_if_missing(NCBI_HS_GENE_INFO_URL, dest)

    entrez_to_symbol: dict[str, str] = {}
    ensembl_to_symbol: dict[str, str] = {}
    with gzip.open(dest, "rt", encoding="utf-8", errors="replace") as f:
        header = next(f, None)
        if header is None:
            raise RuntimeError(f"Empty gene_info file: {dest}")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            gene_id = parts[1].strip()
            symbol = parts[2].strip()
            dbx = parts[5].strip()
            if gene_id and symbol and symbol != "-":
                entrez_to_symbol[gene_id] = symbol
            if dbx and symbol and symbol != "-":
                for x in dbx.split("|"):
                    x = x.strip()
                    if x.startswith("Ensembl:"):
                        ensg = _strip_ensembl_version(x.split(":", 1)[1])
                        if ensg:
                            ensembl_to_symbol[ensg] = symbol

    return GeneIdMaps(entrez_to_symbol=entrez_to_symbol, ensembl_to_symbol=ensembl_to_symbol)


def map_gene_id_to_symbol(raw_id: str, maps: GeneIdMaps) -> str:
    s = (raw_id or "").strip().strip('"')
    if not s:
        return ""
    if s.startswith("ENSG"):
        key = _strip_ensembl_version(s)
        return maps.ensembl_to_symbol.get(key, "")
    if s.isdigit():
        return maps.entrez_to_symbol.get(s, "")
    # Assume already a gene symbol
    return s

