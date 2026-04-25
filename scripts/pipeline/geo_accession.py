from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeoAccession:
    prefix: str
    digits: str

    @property
    def full(self) -> str:
        return f"{self.prefix}{self.digits}"

    @property
    def group_dir(self) -> str:
        # GEO FTP uses <ACC>nnn where last 3 digits are replaced with "nnn"
        # Examples: GSE163211 -> GSE163nnn; GPL11532 -> GPL11nnn; GPL570 -> GPLnnn
        if len(self.digits) <= 3:
            return f"{self.prefix}nnn"
        return f"{self.prefix}{self.digits[:-3]}nnn"


def parse_geo_accession(acc: str) -> GeoAccession:
    acc = (acc or "").strip()
    if len(acc) < 4:
        raise ValueError(f"Invalid GEO accession: {acc!r}")
    prefix = acc[:3]
    digits = acc[3:]
    if prefix not in {"GSE", "GSM", "GPL"}:
        raise ValueError(f"Invalid GEO accession prefix: {acc!r}")
    if not digits.isdigit():
        raise ValueError(f"Invalid GEO accession digits: {acc!r}")
    return GeoAccession(prefix=prefix, digits=digits)

