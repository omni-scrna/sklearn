from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

import warnings

from isomap import load_matrix, run_isomap, select_landmarks, write_geodesics


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

    isomap_embedding, geodesic_distances, landmarks = run_isomap(
        embedding, n_neighbors=2, n_components=2,
    )
    assert landmarks is None

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

    isomap_embedding, geodesic_distances, _ = run_isomap(
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

    _, geodesic_distances, _ = run_isomap(
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

        isomap_embedding, geodesic_distances, _ = run_isomap(
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
        assert h5["geodesic_distances"].dtype == np.float32
        np.testing.assert_allclose(h5["geodesic_distances"][:], distances,
                                   rtol=1e-6)
        assert h5.attrs["tool"] == "sklearn"
        assert int(h5.attrs["n_neighbors"]) == 2
        assert int(h5.attrs["n_components"]) == 2


# --- landmark Isomap (L-Isomap) ------------------------------------------

def _swiss_roll(n=120, seed=0):
    """A 1-D manifold curved through 2-D: geodesic distance along the arc is
    very different from Euclidean distance across the chord, which is what
    Isomap is for.

    A half-circle rather than a spiral: evenly spaced along the curve, so the
    kNN graph stays connected at n_neighbors=5 (a spiral's arms come closer to
    each other than consecutive points do near the centre, which fragments the
    graph). `seed` is accepted for signature stability but the sampling is
    deterministic.
    """
    t = np.linspace(0.0, np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t)])


def test_select_landmarks_min_max_spreads_over_data():
    X = _swiss_roll()
    idx = select_landmarks(X, 10, method="min-max", random_seed=0)

    assert idx.shape == (10,)
    assert len(set(idx.tolist())) == 10          # no duplicates
    assert np.array_equal(idx, np.sort(idx))     # sorted
    # min-max is farthest-point sampling, so it must reach both ends of the
    # curve; random selection frequently would not.
    assert idx.min() < 12 and idx.max() > len(X) - 12


def test_select_landmarks_random_is_seed_reproducible():
    X = _swiss_roll()
    a = select_landmarks(X, 8, method="random", random_seed=42)
    b = select_landmarks(X, 8, method="random", random_seed=42)
    c = select_landmarks(X, 8, method="random", random_seed=7)

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_select_landmarks_accepts_sparse_input():
    X = _swiss_roll(n=40)
    idx = select_landmarks(sp.csr_matrix(X), 6, random_seed=0)
    expected = select_landmarks(X, 6, random_seed=0)

    np.testing.assert_array_equal(idx, expected)


def test_select_landmarks_rejects_bad_count():
    X = _swiss_roll(n=20)
    for bad in (0, -1, 21):
        try:
            select_landmarks(X, bad)
        except ValueError:
            continue
        raise AssertionError(f"n_landmarks={bad} should have raised")


def test_landmark_isomap_returns_narrow_geodesics():
    X = _swiss_roll()

    embedding, geodesics, landmarks = run_isomap(
        X, n_neighbors=5, n_components=2, n_landmarks=15, random_seed=0,
    )

    # The whole point: every cell keeps a row, columns are landmarks only.
    assert embedding.shape == (len(X), 2)
    assert geodesics.shape == (len(X), 15)
    assert landmarks.shape == (15,)
    assert np.isfinite(embedding).all()
    assert np.isfinite(geodesics).all()
    # A landmark's geodesic distance to itself is 0.
    np.testing.assert_allclose(
        geodesics[landmarks, np.arange(15)], np.zeros(15), atol=1e-12
    )


def test_landmark_isomap_accepts_sparse_input():
    X = _swiss_roll(n=60)

    embedding, geodesics, landmarks = run_isomap(
        sp.csr_matrix(X), n_neighbors=5, n_components=2, n_landmarks=10,
        random_seed=0,
    )

    assert embedding.shape == (60, 2)
    assert geodesics.shape == (60, 10)
    assert np.isfinite(embedding).all()


def test_landmark_isomap_approximates_exact_isomap():
    """With every cell a landmark, L-Isomap is the exact algorithm. The
    embeddings need not be sign- or rotation-identical, so compare the
    pairwise distance structure they induce."""
    X = _swiss_roll(n=60)

    exact, _, none_landmarks = run_isomap(X, n_neighbors=5, n_components=2)
    full, _, all_landmarks = run_isomap(
        X, n_neighbors=5, n_components=2, n_landmarks=60, random_seed=0,
    )

    assert none_landmarks is None
    assert all_landmarks.shape == (60,)

    from scipy.spatial.distance import pdist
    np.testing.assert_allclose(pdist(exact), pdist(full), rtol=1e-6, atol=1e-6)


def test_landmark_isomap_preserves_manifold_order():
    """The swiss roll is a 1-D manifold sampled in order, so a good 1-D
    embedding must be monotonic in the sample index (up to sign)."""
    X = _swiss_roll(n=100)

    embedding, _, _ = run_isomap(
        X, n_neighbors=5, n_components=1, n_landmarks=20, random_seed=0,
    )

    order = np.argsort(embedding[:, 0])
    forward = np.array_equal(order, np.arange(100))
    backward = np.array_equal(order, np.arange(99, -1, -1))
    assert forward or backward


def test_landmark_isomap_raises_on_disconnected_graph():
    """DEPARTURE from PR #5969, which warns and zero-fills. Zero means
    'same cell', so that silently collapses unreachable cells together."""
    X = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0],
         [100.0, 0.0], [100.1, 0.0], [100.2, 0.0], [100.3, 0.0]],
    )

    try:
        run_isomap(X, n_neighbors=2, n_components=2, n_landmarks=4,
                   random_seed=0)
    except ValueError as e:
        assert "connected components" in str(e)
        return
    raise AssertionError("disconnected kNN graph should have raised")


def test_write_geodesics_records_landmarks(tmp_path: Path):
    cell_ids = ["cell_a", "cell_b", "cell_c"]
    distances = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 0.0]], dtype=np.float64)
    out = tmp_path / "fixture_geodesics.h5"

    write_geodesics(cell_ids, distances, out, n_neighbors=2, n_components=2,
                    landmarks=np.array([0, 2]))

    with h5py.File(out, "r") as h5:
        assert set(h5.keys()) == {"cell_ids", "geodesic_distances", "landmarks"}
        np.testing.assert_array_equal(h5["landmarks"][:], [0, 2])
        # columns are landmark cells: column j is cell_ids[landmarks[j]]
        assert h5["geodesic_distances"].shape == (3, 2)


def test_write_geodesics_marks_exact_run(tmp_path: Path):
    cell_ids = ["cell_a", "cell_b"]
    distances = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    out = tmp_path / "fixture_geodesics.h5"

    write_geodesics(cell_ids, distances, out, n_neighbors=1, n_components=2)

    with h5py.File(out, "r") as h5:
        # no /landmarks is how a consumer tells exact from landmark runs
        assert "landmarks" not in h5
        assert h5["geodesic_distances"].shape == (2, 2)


def test_write_geodesics_survives_float32_roundtrip(tmp_path: Path):
    """Distances are stored float32; check that costs no meaningful precision
    on values with a realistic magnitude and fractional part."""
    rng = np.random.RandomState(0)
    distances = rng.uniform(0, 45, size=(40, 12)).astype(np.float64)
    out = tmp_path / "fixture_geodesics.h5"

    write_geodesics([f"c{i}" for i in range(40)], distances, out,
                    n_neighbors=5, n_components=2,
                    landmarks=np.arange(12))

    with h5py.File(out, "r") as h5:
        got = h5["geodesic_distances"][:]
    assert np.abs(got - distances).max() < 1e-4


def test_write_geodesics_is_compressed_and_row_chunked(tmp_path: Path):
    distances = np.zeros((200, 40), dtype=np.float64)
    out = tmp_path / "fixture_geodesics.h5"

    write_geodesics([f"c{i}" for i in range(200)], distances, out,
                    n_neighbors=5, n_components=2, landmarks=np.arange(40))

    with h5py.File(out, "r") as h5:
        d = h5["geodesic_distances"]
        assert d.compression == "gzip"
        assert d.chunks == (1, 40)   # one row per chunk
