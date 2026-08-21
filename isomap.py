#!/usr/bin/env python3
"""Experimental Isomap validation module for OmniBenchmark geometry."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.manifold import Isomap


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an experimental Isomap validation on a PCA representation."
    )
    parser.add_argument(
        "--pcas_tsv",
        type=Path,
        required=True,
        help="PCA embedding TSV with cell IDs in the first column.",
    )
    return parser.parse_args()


def read_pca_embedding(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"PCA file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")

    if len(header) < 2 or header[0] != "cell_id":
        raise ValueError(
            "Expected PCA TSV header to start with 'cell_id' "
            "followed by at least one PCA dimension."
        )

    data = np.loadtxt(path, delimiter="\t", skiprows=1, dtype=str, ndmin=2)

    if data.shape[1] != len(header):
        raise ValueError(
            f"PCA TSV has {len(header)} header columns but "
            f"{data.shape[1]} data columns."
        )

    cell_ids = data[:, 0].tolist()
    embedding = data[:, 1:].astype(np.float64)

    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("PCA TSV contains duplicate cell IDs.")

    if not np.isfinite(embedding).all():
        raise ValueError("PCA embedding contains non-finite values.")

    return cell_ids, embedding
def run_isomap(
    embedding: np.ndarray,
    n_neighbors: int = 15,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if embedding.ndim != 2:
        raise ValueError("Input embedding must be a two-dimensional matrix.")

    if embedding.shape[0] <= n_neighbors:
        raise ValueError(
            "n_neighbors must be smaller than the number of cells."
        )

    model = Isomap(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="euclidean",
        path_method="D",
    )

    isomap_embedding = model.fit_transform(embedding)
    geodesic_distances = model.dist_matrix_

    return (
        np.asarray(isomap_embedding, dtype=np.float64),
        np.asarray(geodesic_distances, dtype=np.float64),
    )
def main():
    args = parse_args()
    cell_ids, embedding = read_pca_embedding(args.pcas_tsv)

    print(f"Loaded {len(cell_ids)} cells.")
    print(f"PCA embedding shape: {embedding.shape}")


if __name__ == "__main__":
    main()