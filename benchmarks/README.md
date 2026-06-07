# Benchmarks

This folder contains benchmark scripts for the `mlx-cpt-plus` CPT+ model.

## SPMF Bible benchmark

`benchmark_spmf_bible.py` downloads the SPMF Bible dataset and runs a training + prediction benchmark using the CPT+ model.

Usage:

```bash
python3 benchmarks/benchmark_spmf_bible.py
```

To download only without running the benchmark:

```bash
python3 benchmarks/benchmark_spmf_bible.py --download-only
```

The script stores downloaded datasets under `benchmarks/datasets` by default.
