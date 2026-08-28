#!/usr/bin/env python3
"""Isomap module (sklearn-backed) for omnibenchmark.

Sibling of the PCA stage, not downstream of it: this module reads the same
normalized_selected_h5 matrix PCA does and produces the same embedding_tsv
output contract, so EMBED-M/NNG/GEOM-M can consume either interchangeably
(the ISOMAP stage declares embedding_tsv as an output, same id as PCA's).

Output
------
File: {output_dir}/{name}_embedding.tsv
  Same on-disk shape as scanpy/scrapper's PCA embedding TSV: header
  `cell_id  dim_1  ...  dim_{n_components}`, one row per cell. Columns are named dim_* rather than PC*.

File: {output_dir}/{name}_geodesics.h5
  Flat HDF5 (see write_geodesics below). The pairwise geodesic distance
  matrix is genuinely dense (Isomap's shortest-path step fills in a distance
  for every reachable pair, not just neighbors), so it is stored dense --
  storing it sparse would misrepresent the data. The *input* load stays
  sparse (see load_matrix), matching the source matrix's own format.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
from sklearn.manifold import Isomap

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` (src/common) + module-local writers
from common import cli  # noqa: E402
from writers import Embedding, write_embeddings  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Isomap module (sklearn-backed)")
    cli.add_base_args(p)
    cli.add_stage_args(p, "ISOMAP")
    p.add_argument("--n_neighbors", type=int, required=True,
                   help="Number of neighbors for the local kNN graph")
    p.add_argument("--n_components", type=int, required=True,
                   help="Number of Isomap embedding dimensions")
    return p.parse_args()


def load_matrix(h5_path) -> tuple[list[str], sp.csr_matrix]:
    """Read the normalized, gene-selected matrix as a sparse cells-by-genes
    CSR array. Same on-disk layout and orientation fix as scanpy/pca.py's
    load_matrix: the file stores genes-by-cells CSC, transposed here."""
    with h5py.File(h5_path, "r") as h5:
        g = h5["matrix"]
        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        shape = tuple(g["shape"][:])
        cell_ids = g["barcodes"][:].astype(str)

    X = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()  # cells x genes
    return list(cell_ids), X


def run_isomap(
    embedding,
    n_neighbors: int = 15,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Isomap on `embedding` (dense ndarray or sparse array, cells x
    features) and return (isomap_embedding, geodesic_distances)."""
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


def write_geodesics(cell_ids, geodesic_distances, path, n_neighbors, n_components):
    """Dense HDF5 geodesic distance matrix, root attrs describe how it was
    computed. Not CSR: dist_matrix_ is a full pairwise matrix, not sparse
    data, so it's stored as a plain dense dataset."""
    with h5py.File(path, "w") as h5:
        # dtype="S": h5py can't write numpy unicode ('<U') arrays; bytes give
        # portable fixed-length HDF5 strings (matches knn.py's convention).
        h5.create_dataset("cell_ids", data=np.array(cell_ids, dtype="S"))
        h5.create_dataset("geodesic_distances", data=geodesic_distances, dtype="float64")
        h5.attrs["tool"] = "sklearn"
        h5.attrs["n_neighbors"] = n_neighbors
        h5.attrs["n_components"] = n_components


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "normalized_selected_h5", "n_neighbors", "n_components"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cell_ids, X = load_matrix(args.normalized_selected_h5)

    embedding, geodesic_distances = run_isomap(
        X, n_neighbors=args.n_neighbors, n_components=args.n_components,
    )

    col_names = [f"dim_{i + 1}" for i in range(embedding.shape[1])]
    embedding_out = Path(args.output_dir) / f"{args.name}_embedding.tsv"
    write_embeddings(Embedding(embedding, cell_ids, col_names), embedding_out)
    print(f"  wrote: {embedding_out}")

    geodesics_out = Path(args.output_dir) / f"{args.name}_geodesics.h5"
    write_geodesics(cell_ids, geodesic_distances, geodesics_out,
                     args.n_neighbors, args.n_components)
    print(f"  wrote: {geodesics_out}")


if __name__ == "__main__":
    main()
