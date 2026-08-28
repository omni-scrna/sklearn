from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

import warnings

from isomap import load_matrix, run_isomap, write_geodesics


def _write_matrix_h5(path: Path, X: sp.csr_matrix, cell_ids, gene_ids):
    """Genes-by-cells CSC on disk, matching scanpy/pca.py's load_matrix
    contract (normalized_selected_h5)."""
    Xt = X.T.tocsc()  # genes x cells
    with h5py.File(path, "w") as h5:
        g = h5.create_group("matrix")
        g.create_dataset("data", data=Xt.data)
        g.create_dataset("indices", data=Xt.indices)
        g.create_dataset("indptr", data=Xt.indptr)
        g.create_dataset("shape", data=np.array(Xt.shape))
        g.create_dataset("genes", data=np.array(gene_ids, dtype="S"))
        g.create_dataset("barcodes", data=np.array(cell_ids, dtype="S"))


def test_load_matrix_reads_cells_by_genes_sparse(tmp_path: Path):
    X = sp.csr_matrix(np.array(
        [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]
    ))
    h5_path = tmp_path / "fixture_normalized_selected.h5"
    _write_matrix_h5(h5_path, X, ["cell_a", "cell_b", "cell_c"], ["gene_1", "gene_2", "gene_3"])

    cell_ids, loaded = load_matrix(h5_path)

    assert cell_ids == ["cell_a", "cell_b", "cell_c"]
    assert sp.issparse(loaded)
    assert loaded.shape == (3, 3)
    np.testing.assert_allclose(loaded.toarray(), X.toarray())


def test_run_isomap_accepts_sparse_input():
    # Same line-of-points fixture as the dense tests below, sparse this time.
    embedding = sp.csr_matrix(np.array(
        [[float(i), 0.0] for i in range(6)]
    ))

    isomap_embedding, geodesic_distances = run_isomap(
        embedding, n_neighbors=2, n_components=2,
    )

    assert isomap_embedding.shape == (6, 2)
    assert geodesic_distances.shape == (6, 6)
    assert np.isfinite(isomap_embedding).all()
    assert np.isfinite(geodesic_distances).all()


def test_run_isomap_returns_expected_shapes():
    embedding = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
        ],
        dtype=np.float64,
    )

    isomap_embedding, geodesic_distances = run_isomap(
        embedding,
        n_neighbors=2,
        n_components=2,
    )

    assert isomap_embedding.shape == (6, 2)
    assert geodesic_distances.shape == (6, 6)

    assert np.isfinite(isomap_embedding).all()
    assert np.isfinite(geodesic_distances).all()

    np.testing.assert_allclose(
        np.diag(geodesic_distances),
        np.zeros(6),
    )

    np.testing.assert_allclose(
        geodesic_distances,
        geodesic_distances.T,
    )


def test_isomap_geodesic_distances_on_line():
    embedding = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
        ],
        dtype=np.float64,
    )

    _, geodesic_distances = run_isomap(
        embedding,
        n_neighbors=2,
        n_components=2,
    )

    expected = np.abs(
        np.arange(6)[:, None] - np.arange(6)[None, :]
    ).astype(np.float64)

    np.testing.assert_allclose(
        geodesic_distances,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_isomap_handles_disconnected_knn_graph():
    embedding = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [100.0, 0.0],
            [100.1, 0.0],
            [100.2, 0.0],
        ],
        dtype=np.float64,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        isomap_embedding, geodesic_distances = run_isomap(
            embedding,
            n_neighbors=2,
            n_components=2,
        )
        warning_messages = [str(warning.message) for warning in caught_warnings]

    assert any(
        "number of connected components" in message
        and "2 > 1" in message
        for message in warning_messages
    )
    assert isomap_embedding.shape == (6, 2)
    assert geodesic_distances.shape == (6, 6)
    assert np.isfinite(isomap_embedding).all()
    assert np.isfinite(geodesic_distances).all()


def test_write_geodesics_roundtrip(tmp_path: Path):
    cell_ids = ["cell_a", "cell_b", "cell_c"]
    distances = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]], dtype=np.float64
    )
    out = tmp_path / "fixture_geodesics.h5"

    write_geodesics(cell_ids, distances, out, n_neighbors=2, n_components=2)

    with h5py.File(out, "r") as h5:
        assert set(h5.keys()) == {"cell_ids", "geodesic_distances"}
        assert [c.decode() for c in h5["cell_ids"][:]] == cell_ids
        np.testing.assert_allclose(h5["geodesic_distances"][:], distances)
        assert h5.attrs["tool"] == "sklearn"
        assert int(h5.attrs["n_neighbors"]) == 2
        assert int(h5.attrs["n_components"]) == 2
