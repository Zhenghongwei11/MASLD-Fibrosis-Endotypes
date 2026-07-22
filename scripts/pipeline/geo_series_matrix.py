from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


@dataclass(frozen=True)
class SeriesMatrix:
    gse: str
    platform_id: str
    sample_ids: list[str]
    characteristics: dict[str, list[str]]
    table_header: list[str]
    table_rows_path: Path


RE_SERIES_PLATFORM = re.compile(r"^!Series_platform_id\b")
RE_SAMPLE_GSM = re.compile(r"^!Sample_geo_accession\b")
RE_SAMPLE_TITLE = re.compile(r"^!Sample_title\b")
RE_CH = re.compile(r"^!Sample_characteristics_ch1\b")


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def read_series_matrix_header(gz_path: Path) -> tuple[str, list[str], dict[str, list[str]]]:
    platform_id = ""
    sample_ids: list[str] = []
    characteristics: dict[str, list[str]] = {}

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if RE_SERIES_PLATFORM.match(line):
                parts = line.split("\t")[1:]
                vals = [_strip_quotes(p) for p in parts if p.strip()]
                platform_id = ",".join(vals)
                continue
            if RE_SAMPLE_GSM.match(line):
                parts = line.split("\t")[1:]
                sample_ids = [_strip_quotes(p) for p in parts]
                continue
            if line.startswith("!series_matrix_table_begin"):
                break
            if not RE_CH.match(line):
                continue
            parts = line.split("\t")[1:]
            if not parts:
                continue
            parts = [_strip_quotes(p) for p in parts]
            first = parts[0]
            if ":" not in first:
                continue
            key = first.split(":", 1)[0].strip()
            if not key:
                continue
            values: list[str] = []
            for p in parts:
                if ":" in p:
                    values.append(p.split(":", 1)[1].strip())
                else:
                    values.append("")
            if sample_ids and len(values) < len(sample_ids):
                values = values + [""] * (len(sample_ids) - len(values))
            if sample_ids and len(values) > len(sample_ids):
                values = values[: len(sample_ids)]
            characteristics[key] = values

    return platform_id, sample_ids, characteristics


def read_series_matrix_sample_titles(gz_path: Path) -> list[str]:
    """
    Returns the !Sample_title row values in the same order as !Sample_geo_accession.
    Returns [] if not present.
    """
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                break
            if RE_SAMPLE_TITLE.match(line):
                parts = line.split("\t")[1:]
                return [_strip_quotes(p) for p in parts]
    return []


def iter_series_matrix_table(gz_path: Path) -> Iterator[list[str]]:
    """
    Yields raw TSV rows (already unquoted) from the series_matrix_table section, including header.
    """
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            if raw.startswith("!series_matrix_table_begin"):
                break
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row and row[0].startswith("!series_matrix_table_end"):
                break
            yield [_strip_quotes(c) for c in row]


def has_expression_table(gz_path: Path) -> bool:
    for row in iter_series_matrix_table(gz_path):
        if not row:
            continue
        if row[0] == "ID_REF":
            continue
        return True
    return False


def read_expression_matrix(
    gz_path: Path,
    *,
    max_genes: int | None = None,
) -> tuple[list[str], list[str], list[list[float]]]:
    """
    Returns (feature_ids, sample_ids, values) where values is rows=features, cols=samples.
    Expects a normal series matrix table with numeric values.
    """
    it = iter_series_matrix_table(gz_path)
    header = next(it, None)
    if not header or header[0] != "ID_REF":
        raise RuntimeError(f"Missing expression header in {gz_path}")
    sample_ids = header[1:]

    feature_ids: list[str] = []
    values: list[list[float]] = []
    for row in it:
        if not row or not row[0]:
            continue
        if row[0] == "ID_REF":
            continue
        feature_ids.append(row[0])
        vals: list[float] = []
        for x in row[1:]:
            try:
                vals.append(float(x))
            except Exception:
                vals.append(float("nan"))
        values.append(vals)
        if max_genes is not None and len(feature_ids) >= max_genes:
            break

    return feature_ids, sample_ids, values
