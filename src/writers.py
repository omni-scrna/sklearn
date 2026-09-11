"""Reusable writers for objects produced by this module, conforming to the omni-scrna benchmark spec.

Trimmed copy of scanpy's src/writers.py: this module only ever emits an
embedding (no loadings), so Loadings/write_loadings aren't carried over.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class Embedding:
    matrix: np.ndarray        # shape (n_cells, n_dims)
    row_ids: list             # cell barcodes, length n_cells
    col_names: list = field(default_factory=list)  # dim labels; auto-generated if empty


def _col_names(embedding):
    if embedding.col_names:
        return embedding.col_names
    return [f"dim_{i + 1}" for i in range(embedding.matrix.shape[1])]


def _write_tsv(path, embedding, row_label):
    # Header has N cols, data rows N+1 (row_ids unnamed first column);
    # read.table(f, header=TRUE) auto-promotes the extra leading column to row.names.
    cols = _col_names(embedding)
    df = pl.from_numpy(embedding.matrix, schema=cols).insert_column(
        0, pl.Series("", embedding.row_ids))
    with open(path, "w") as f:
        f.write(row_label + "\t" + "\t".join(cols) + "\n")
        df.write_csv(f, separator="\t", include_header=False)


def write_embeddings(obj, path, format="tsv"):
    if format == "tsv":
        _write_tsv(path, obj, "cell_id")
    else:
        raise ValueError(f"unsupported format: {format!r}")
