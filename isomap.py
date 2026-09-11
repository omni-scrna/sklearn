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
  Dense HDF5, (n_cells, n_sources), labelled on both axes -- see
  write_geodesics. The input load stays sparse (see load_matrix).

Landmark Isomap
---------------
Exact Isomap is O(n^2) in cells. --n_landmarks switches to L-Isomap (de Silva
& Tenenbaum): Dijkstra from L landmark cells instead of all n, giving an
(n, L) matrix and an L x L eigenproblem. Follows scikit-learn PR #5969, which
was reviewed but never merged. Two deliberate departures from that
PR are marked DEPARTURE below.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial.distance import cdist
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import check_random_state

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
    # Hand-rolled, not schema-driven: schema args are all required=True, and
    # --n_landmarks must be omittable to select exact Isomap.
    p.add_argument("--n_landmarks", type=int, default=None,
                   help="Landmark cells for L-Isomap; omitted runs exact "
                        "Isomap. Should exceed n_components comfortably "
                        "(L == n_cells is equivalent to exact).")
    p.add_argument("--landmark_method", choices=["min-max", "random"],
                   default="min-max",
                   help="Landmark selection; min-max is greedy farthest-point "
                        "sampling. Only used with --n_landmarks.")
    p.add_argument("--random_seed", type=int, default=None,
                   help="Seed for landmark selection. Only used with "
                        "--n_landmarks: exact Isomap has no stochastic step.")
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


def select_landmarks(X, n_landmarks: int, method: str = "min-max",
                     random_seed: int | None = None) -> np.ndarray:
    """Indices of `n_landmarks` cells to use as shortest-path sources.

    'min-max' is greedy farthest-point sampling: repeatedly take the cell
    farthest from every landmark chosen so far. Spreads landmarks over the
    manifold rather than clumping them where cells are dense, at no measured
    cost over random.

    Runs in feature space, not on the kNN graph, so a sparse X is densified --
    affordable at the gene-selected width this stage receives (~2000 genes).
    """
    n_cells = X.shape[0]
    if not 0 < n_landmarks <= n_cells:
        raise ValueError(
            f"n_landmarks must be in 1..n_cells ({n_cells}), got {n_landmarks}."
        )

    rng = check_random_state(random_seed)

    if method == "random":
        return np.sort(rng.choice(n_cells, size=n_landmarks, replace=False))

    if method != "min-max":
        raise ValueError(f"Unrecognized landmark method '{method}'.")

    Xd = X.toarray() if sp.issparse(X) else np.asarray(X)
    landmarks = [int(rng.randint(n_cells))]
    # nearest-landmark distance for every cell, updated as landmarks are added
    nearest = cdist(Xd[landmarks], Xd).ravel()
    for _ in range(1, n_landmarks):
        pick = int(np.argmax(nearest))
        landmarks.append(pick)
        nearest = np.minimum(nearest, cdist(Xd[pick, None], Xd).ravel())

    return np.sort(np.array(landmarks))


def run_isomap(
    embedding,
    n_neighbors: int = 15,
    n_components: int = 2,
    n_landmarks: int | None = None,
    landmark_method: str = "min-max",
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Fit Isomap on `embedding` (dense ndarray or sparse array, cells x
    features) and return (isomap_embedding, geodesic_distances, landmarks).

    With n_landmarks=None this is exact Isomap and `landmarks` is None;
    geodesic_distances is (n_cells, n_cells). With n_landmarks=L it is
    L-Isomap: geodesic_distances is (n_cells, L) and `landmarks` holds the
    indices of the source cells.
    """
    if embedding.ndim != 2:
        raise ValueError("Input embedding must be a two-dimensional matrix.")

    if embedding.shape[0] <= n_neighbors:
        raise ValueError(
            "n_neighbors must be smaller than the number of cells."
        )

    if n_landmarks is None:
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
            None,
        )

    return _run_landmark_isomap(
        embedding, n_neighbors, n_components, n_landmarks,
        landmark_method, random_seed,
    )


def _run_landmark_isomap(embedding, n_neighbors, n_components, n_landmarks,
                         landmark_method, random_seed):
    """L-Isomap: full kNN graph, Dijkstra from landmarks only, classical MDS
    on the (L, L) block, then project every cell into that frame. Follows
    scikit-learn PR #5969's fit(), except where marked DEPARTURE.
    """
    graph = kneighbors_graph(embedding, n_neighbors, mode="distance")

    # DEPARTURE: PR #5969 warns on a disconnected graph and zero-fills the
    # infinities. 0 means "same cell", so unreachable cells would be embedded
    # on top of each other -- and this benchmark's datasets are the ones at
    # risk (be1 is 8 cell lines, cb is cross-species).
    n_parts, _ = connected_components(graph, directed=False)
    if n_parts > 1:
        raise ValueError(
            f"kNN graph has {n_parts} connected components; geodesic "
            f"distances between them are undefined. Increase n_neighbors "
            f"(currently {n_neighbors})."
        )

    landmarks = select_landmarks(embedding, n_landmarks, landmark_method,
                                 random_seed)

    # The memory win: (L, n) rather than (n, n), transposed so cells are rows.
    geodesic_distances = dijkstra(graph, directed=False, indices=landmarks).T

    # Classical MDS on the (L, L) landmark block -> the coordinate frame.
    G = geodesic_distances[landmarks] ** 2
    G *= -0.5
    kernel_pca = KernelPCA(n_components=n_components, kernel="precomputed")
    kernel_pca.fit(G)

    # Triangulate every cell into that frame from its landmark distances.
    # KernelPCA.transform is the Nystrom projection sklearn's own
    # Isomap.transform() uses out-of-sample, just against L columns not n.
    G_all = geodesic_distances ** 2
    G_all *= -0.5
    isomap_embedding = kernel_pca.transform(G_all)

    return (
        np.asarray(isomap_embedding, dtype=np.float64),
        np.asarray(geodesic_distances, dtype=np.float64),
        landmarks,
    )


def write_geodesics(cell_ids, geodesic_distances, path, n_neighbors,
                    n_components, landmarks=None):
    """Dense HDF5 geodesic distance matrix, labelled on both axes.

        /cell_ids            (n,)      bytes     row labels, input order
        /geodesic_distances  (n, cols) float32
        /landmarks           (L,)      int64     column labels; landmark runs only

    Columns are the shortest-path sources: every cell for exact Isomap (in
    cell_ids order), the landmarks for L-Isomap, where column j is the cell at
    cell_ids[landmarks[j]]. `"landmarks" in h5` tests which.

    Dense, not CSR: geodesics are ~100% nonzero, so CSR costs ~1.5x. float32
    is well beyond the precision these distances carry (max relative
    round-trip error ~6e-8) and halves the file.
    """
    matrix = np.asarray(geodesic_distances)
    with h5py.File(path, "w") as h5:
        # dtype="S": h5py can't write numpy unicode ('<U') arrays (knn.py does the same).
        h5.create_dataset("cell_ids", data=np.array(cell_ids, dtype="S"))
        h5.create_dataset(
            "geodesic_distances", data=matrix, dtype="float32",
            chunks=(1, matrix.shape[1]),  # row chunks: ~10x faster per-cell reads
            compression="gzip", compression_opts=4,
        )
        if landmarks is not None:
            h5.create_dataset("landmarks", data=np.asarray(landmarks, dtype="int64"))
        h5.attrs["tool"] = "sklearn"
        h5.attrs["n_neighbors"] = n_neighbors
        h5.attrs["n_components"] = n_components


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "normalized_selected_h5", "n_neighbors",
              "n_components", "n_landmarks", "landmark_method", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cell_ids, X = load_matrix(args.normalized_selected_h5)

    embedding, geodesic_distances, landmarks = run_isomap(
        X, n_neighbors=args.n_neighbors, n_components=args.n_components,
        n_landmarks=args.n_landmarks, landmark_method=args.landmark_method,
        random_seed=args.random_seed,
    )
    if landmarks is None:
        print(f"  exact Isomap: geodesics {geodesic_distances.shape}")
    else:
        print(f"  L-Isomap on {len(landmarks)} landmarks "
              f"({args.landmark_method}): geodesics {geodesic_distances.shape}")

    col_names = [f"dim_{i + 1}" for i in range(embedding.shape[1])]
    embedding_out = Path(args.output_dir) / f"{args.name}_embedding.tsv"
    write_embeddings(Embedding(embedding, cell_ids, col_names), embedding_out)
    print(f"  wrote: {embedding_out}")

    geodesics_out = Path(args.output_dir) / f"{args.name}_geodesics.h5"
    write_geodesics(cell_ids, geodesic_distances, geodesics_out,
                     args.n_neighbors, args.n_components, landmarks)
    print(f"  wrote: {geodesics_out}")


if __name__ == "__main__":
    main()
