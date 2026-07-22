from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    sha256: str
    nbytes: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_if_missing(url: str, dest: Path, *, timeout_s: int = 60) -> DownloadResult:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout_s) as r, dest.open("wb") as f:  # nosec - public URL
            shutil.copyfileobj(r, f)
    nbytes = dest.stat().st_size
    return DownloadResult(path=dest, sha256=sha256_file(dest), nbytes=nbytes)
