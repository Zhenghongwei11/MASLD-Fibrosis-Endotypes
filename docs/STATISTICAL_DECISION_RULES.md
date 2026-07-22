# Statistical decision rules

- Endpoint: advanced fibrosis defined as histologic fibrosis stage ≥F3 vs ≤F2 where staging is available.
- Labels whose reported ranges cross the F3 threshold are treated as missing for the primary endpoint.
- GSE163211 is used for targeted-panel NMF discovery and is not included in the primary F3+ validation endpoint.
- Effect sizes: within-cohort logistic regression odds ratio (OR) per 1-SD transferred transcriptomic-axis score.
- Synthesis: REML random-effects meta-analysis with Hartung-Knapp confidence intervals; DerSimonian-Laird and iterated random-effects estimates are retained as sensitivity analyses.
- Transportability: leave-one-cohort-out (LOCO) evaluation; discrimination reported as AUC, calibration with calibration slope and reliability plots, and probability accuracy with Brier score.
- Benchmarking: published MASLD/MASH/fibrosis signatures are scored using the same within-cohort standardized mean-expression rule.
- Annotation: pathway enrichment uses the discovery-platform gene universe; single-cell RNA-seq localization is used for cell-compartment context.
- Robustness: NMF rank checks are reported descriptively and are not used for post-hoc selection of the primary model.
