# MASLD Fibrosis Endotypes (Public Reproducibility Package)

This repository provides a reproducible pipeline to derive liver transcriptomic endotypes and evaluate
their association with advanced fibrosis across multiple public GEO liver-biopsy cohorts.

## Quick start (one command)

```bash
bash scripts/reproduce_one_click.sh
```

## What the pipeline does

- Downloads required public inputs on demand (GEO series matrices; selected GEO supplementary files; NCBI gene info; GPL annotations).
- Extracts fibrosis staging metadata and defines advanced fibrosis as stage ≥F3 (vs ≤F2) where available.
- Discovers endotypes in one cohort using non-negative matrix factorization (NMF) and defines compact gene signatures.
- Transfers signatures across cohorts with within-cohort standardized scoring (and reports coverage).
- Estimates cohort-wise odds ratios per 1-SD endotype score and pools by random-effects meta-analysis.
- Benchmarks the transferred endotype against published MASLD/MASH/fibrosis transcriptomic signatures.
- Evaluates transportability with leave-one-cohort-out prediction, emphasizing calibration and decision-curve analysis.
- Annotates endotype signatures with pathway enrichment and an external human liver scRNA atlas.

## Outputs

- Derived tables: `results/`
- Publication-ready figures: `plots/publication/`
- Provenance map: `docs/FIGURE_PROVENANCE.tsv`

## System requirements

- Python 3.10+
- macOS/Linux recommended

## Notes

- First run may take time due to public downloads.
- No raw private data is included; all inputs are downloaded from public sources listed in `docs/DATA_MANIFEST.tsv`.
