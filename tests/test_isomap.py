from pathlib import Path

import numpy as np

from isomap import read_pca_embedding, run_isomap

def test_read_pca_embedding(tmp_path: Path):
    pca_path = tmp_path / "fixture_pcas.tsv"
    pca_path.write_text(
        "cell_id\tPC1\tPC2\n"
        "cell_a\t1.0\t2.0\n"
        "cell_b\t3.0\t4.0\n"
        "cell_c\t5.0\t6.0\n",
        encoding="utf-8",
    )

    cell_ids, embedding = read_pca_embedding(pca_path)

    assert cell_ids == ["cell_a", "cell_b", "cell_c"]
    assert embedding.shape == (3, 2)
    assert embedding.dtype == np.float64
    np.testing.assert_allclose(
        embedding,
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ),
    )

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