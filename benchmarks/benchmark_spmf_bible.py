#!/usr/bin/env python3
"""Benchmark CPT+ on the SPMF Bible sequence dataset."""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from mlx_cpt_plus.model import CPTPlus

SPMF_DATASET_NAME = "BIBLE.txt"
SPMF_BASE_URL = "https://www.philippe-fournier-viger.com/spmf/publicdatasets"


def download_spmf_dataset(
    target_dir: Path,
    dataset_url: str,
    dataset_name: str = SPMF_DATASET_NAME,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = target_dir / dataset_name
    if dataset_path.exists():
        return dataset_path

    url = dataset_url
    print(f"Downloading SPMF dataset from {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
        dataset_path.write_bytes(data)
        print(f"Saved dataset to {dataset_path}")
    except Exception as exc:
        raise RuntimeError(
            f"Unable to download dataset from {url}: {exc}"
        ) from exc

    return dataset_path


def load_spmf_sequences(dataset_path: Path) -> List[List[int]]:
    sequences: List[List[int]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith("#"):
                continue

            tokens = []
            for token in clean_line.split():
                if token in {"-1", "-2"}:
                    continue
                tokens.append(int(token))

            if tokens:
                sequences.append(tokens)

    if not sequences:
        raise ValueError(f"No sequences loaded from {dataset_path}")

    return sequences


def benchmark_training(sequences: List[List[int]], max_context_length: int = 10) -> None:
    print("Building CPT+ model")
    model = CPTPlus()
    start = time.perf_counter()
    model.fit(sequences)
    fit_duration = time.perf_counter() - start

    print(f"Fit completed in {fit_duration:.3f} seconds")
    print(f"Sequences: {len(sequences)}")
    print(f"Vocabulary size: {len(model.vocabulary)}")

    sample_contexts = []
    for sequence in sequences[-50:]:
        if len(sequence) < 2:
            continue
        start_idx = max(0, len(sequence) - max_context_length - 1)
        sample_contexts.append(sequence[start_idx:-1])

    if not sample_contexts:
        print("No contexts available for prediction benchmarking.")
        return

    print(f"Benchmarking prediction on {len(sample_contexts)} contexts")
    prediction_start = time.perf_counter()
    for context in sample_contexts:
        model.predict(context, k=5)
    prediction_duration = time.perf_counter() - prediction_start

    avg_prediction = prediction_duration / len(sample_contexts)
    print(f"Total prediction time: {prediction_duration:.3f} seconds")
    print(f"Average prediction latency: {avg_prediction * 1000:.3f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CPT+ on the SPMF Bible dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        default="benchmarks/datasets",
        help="Directory to store SPMF datasets.",
    )
    parser.add_argument(
        "--dataset-name",
        default=SPMF_DATASET_NAME,
        help="SPMF dataset filename.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download the SPMF dataset and exit without running the benchmark.",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=10,
        help="Maximum context length used for prediction benchmarking.",
    )
    parser.add_argument(
        "--dataset-url",
        default=f"{SPMF_BASE_URL}/{SPMF_DATASET_NAME}",
        help="URL to download the SPMF dataset from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    dataset_path = download_spmf_dataset(
        dataset_dir, args.dataset_url, args.dataset_name
    )

    if args.download_only:
        print("Dataset download finished.")
        return

    sequences = load_spmf_sequences(dataset_path)
    print(f"Loaded {len(sequences)} sequences from {dataset_path}")
    benchmark_training(sequences, max_context_length=args.max_context_length)


if __name__ == "__main__":
    main()
