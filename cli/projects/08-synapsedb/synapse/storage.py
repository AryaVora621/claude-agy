"""
SynapseDB: Columnar Table Storage and Parquet-Style Binary Format.
Features:
1. ColumnarTable chunked into row groups of fixed batch size (e.g. 1024 rows).
2. Zone Map predicate pruning: skips non-matching row groups before reading column vectors.
3. Projection pruning: reads and materializes only queried columns.
4. Binary on-disk persistence with metadata footer and lazy chunk indexing.
"""

import os
import struct
import json
from typing import List, Dict, Tuple, Optional, Any, Iterator, Sequence
from synapse.types import DataType, Vector, RecordBatch, SelectionVector
from synapse.encoding import ZoneMap, RLECodec, DictionaryCodec, BitPackingCodec

MAGIC_BYTES = b"SYNAPSE1"
DEFAULT_CHUNK_SIZE = 1024


class TableSchema:
    """Table schema defining column names, data types, and nullability."""
    __slots__ = ("columns", "types", "_col_idx")

    def __init__(self, columns: Dict[str, DataType]):
        self.columns = list(columns.keys())
        self.types = columns
        self._col_idx = {name: idx for idx, name in enumerate(self.columns)}

    def has_column(self, name: str) -> bool:
        return name in self.types

    def get_type(self, name: str) -> DataType:
        return self.types[name]

    def to_dict(self) -> Dict[str, str]:
        return {name: dt.name for name, dt in self.types.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "TableSchema":
        types = {name: DataType[dt_str] for name, dt_str in d.items()}
        return cls(types)


class RowGroup:
    """
    Horizontal partition of a columnar table containing equal-sized column vectors.
    Accompanied by Zone Maps for each column to support chunk-level predicate pruning.
    """
    __slots__ = ("columns", "zone_maps", "row_count")

    def __init__(self, columns: Dict[str, Vector]):
        self.columns = columns
        if columns:
            first_col = next(iter(columns.values()))
            self.row_count = len(first_col)
        else:
            self.row_count = 0

        self.zone_maps: Dict[str, ZoneMap] = {}
        for name, col in columns.items():
            self.zone_maps[name] = ZoneMap.compute(col)

    def can_match_predicates(self, predicates: Sequence[Tuple[str, str, Any]]) -> bool:
        """Evaluate if this entire row group can be pruned via Zone Maps."""
        for col_name, op, target_val in predicates:
            zm = self.zone_maps.get(col_name)
            if zm is not None:
                if not zm.can_match(op, target_val):
                    return False  # Prune this row group!
        return True

    def project(self, column_names: Sequence[str]) -> RecordBatch:
        projected = {name: self.columns[name] for name in column_names}
        return RecordBatch(projected)

    def to_batch(self) -> RecordBatch:
        return RecordBatch(dict(self.columns))


class ColumnarTable:
    """
    In-memory and disk-backed vectorized columnar table.
    Stores data in horizontal row groups with zone map metadata.
    """

    def __init__(self, name: str, schema: TableSchema, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.name = name
        self.schema = schema
        self.chunk_size = chunk_size
        self.row_groups: List[RowGroup] = []
        self.total_rows = 0

    def insert_batch(self, batch: RecordBatch) -> None:
        """Append a RecordBatch, slicing into row groups if necessary."""
        for col_name in self.schema.columns:
            if col_name not in batch.columns:
                raise ValueError(f"Batch missing required column: {col_name}")

        offset = 0
        total = len(batch)
        while offset < total:
            count = min(self.chunk_size, total - offset)
            sub_batch = batch.slice(offset, count)
            rg = RowGroup(dict(sub_batch.columns))
            self.row_groups.append(rg)
            offset += count

        self.total_rows += total

    def insert_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Convert row dictionaries into a columnar RecordBatch and append."""
        if not rows:
            return

        cols: Dict[str, Vector] = {}
        for col_name in self.schema.columns:
            dt = self.schema.get_type(col_name)
            col_vals = [r.get(col_name) for r in rows]
            cols[col_name] = Vector(dt, col_vals)

        batch = RecordBatch(cols)
        self.insert_batch(batch)

    def scan(
        self,
        projection: Optional[Sequence[str]] = None,
        predicates: Optional[Sequence[Tuple[str, str, Any]]] = None
    ) -> Iterator[RecordBatch]:
        """
        Scan table row groups with zone map pruning and column projection pruning.
        """
        proj_cols = list(projection) if projection is not None else self.schema.columns
        for c in proj_cols:
            if not self.schema.has_column(c):
                raise KeyError(f"Projected column '{c}' does not exist in schema")

        for rg in self.row_groups:
            # 1. Zone Map Predicate Pruning
            if predicates and not rg.can_match_predicates(predicates):
                continue  # Skip row group entirely!

            # 2. Projection Pruning
            yield rg.project(proj_cols)

    def save_to_file(self, filepath: str) -> None:
        """
        Serialize table to disk in custom Parquet-like binary columnar format.
        Layout:
        [Magic: 8B] [Data Chunks: Variable] [Footer JSON] [Footer Len: 4B] [Magic: 8B]
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        with open(filepath, "wb") as f:
            f.write(MAGIC_BYTES)

            chunk_meta = []
            for rg_idx, rg in enumerate(self.row_groups):
                rg_cols_meta = {}
                for col_name, vec in rg.columns.items():
                    data_offset = f.tell()

                    # Write null bitmap
                    null_bytes = vec.null_bitmap.bytes_data
                    f.write(struct.pack("!I", len(null_bytes)))
                    f.write(null_bytes)

                    # Write vector data
                    if vec.data_type == DataType.INT64:
                        raw_data = vec.data.tobytes()
                    elif vec.data_type == DataType.FLOAT64:
                        raw_data = vec.data.tobytes()
                    elif vec.data_type == DataType.BOOLEAN:
                        raw_data = vec.data.tobytes()
                    else:
                        # Strings encoded as JSON array for serialization
                        raw_data = json.dumps(vec.data).encode("utf-8")

                    f.write(struct.pack("!I", len(raw_data)))
                    f.write(raw_data)

                    total_col_bytes = f.tell() - data_offset
                    zm = rg.zone_maps[col_name]
                    rg_cols_meta[col_name] = {
                        "offset": data_offset,
                        "bytes": total_col_bytes,
                        "min": zm.min_val,
                        "max": zm.max_val,
                        "null_count": zm.null_count,
                        "row_count": zm.row_count
                    }

                chunk_meta.append({
                    "rg_idx": rg_idx,
                    "row_count": rg.row_count,
                    "columns": rg_cols_meta
                })

            footer_data = {
                "name": self.name,
                "schema": self.schema.to_dict(),
                "chunk_size": self.chunk_size,
                "total_rows": self.total_rows,
                "row_groups": chunk_meta
            }

            footer_bytes = json.dumps(footer_data).encode("utf-8")
            f.write(footer_bytes)
            f.write(struct.pack("!I", len(footer_bytes)))
            f.write(MAGIC_BYTES)

    @classmethod
    def load_from_file(cls, filepath: str) -> "ColumnarTable":
        """Deserialize table from disk file."""
        with open(filepath, "rb") as f:
            file_size = f.seek(0, os.SEEK_END)
            if file_size < 20:
                raise ValueError("Corrupt file: size too small")

            # Check footer magic
            f.seek(file_size - 8)
            end_magic = f.read(8)
            if end_magic != MAGIC_BYTES:
                raise ValueError("Invalid file magic trailer")

            # Read footer length
            f.seek(file_size - 12)
            footer_len = struct.unpack("!I", f.read(4))[0]

            # Read footer JSON
            f.seek(file_size - 12 - footer_len)
            footer_json = f.read(footer_len).decode("utf-8")
            meta = json.loads(footer_json)

            schema = TableSchema.from_dict(meta["schema"])
            table = cls(meta["name"], schema, meta["chunk_size"])
            table.total_rows = meta["total_rows"]

            # Reconstruct row groups
            for rg_meta in meta["row_groups"]:
                cols = {}
                for col_name, c_meta in rg_meta["columns"].items():
                    f.seek(c_meta["offset"])
                    dt = schema.get_type(col_name)

                    # Read nulls
                    null_len = struct.unpack("!I", f.read(4))[0]
                    null_bytes = f.read(null_len)

                    # Read data
                    data_len = struct.unpack("!I", f.read(4))[0]
                    raw_data = f.read(data_len)

                    count = c_meta["row_count"]
                    if dt == DataType.INT64:
                        v = Vector(dt, [])
                        v.data.frombytes(raw_data)
                        v.length = count
                        v.null_bitmap.size = count
                    elif dt == DataType.FLOAT64:
                        v = Vector(dt, [])
                        v.data.frombytes(raw_data)
                        v.length = count
                        v.null_bitmap.size = count
                    elif dt == DataType.BOOLEAN:
                        v = Vector(dt, [])
                        v.data.frombytes(raw_data)
                        v.length = count
                        v.null_bitmap.size = count
                    else:
                        str_list = json.loads(raw_data.decode("utf-8"))
                        v = Vector(dt, str_list)

                    v.null_bitmap.bytes_data = bytearray(null_bytes)
                    cols[col_name] = v

                table.row_groups.append(RowGroup(cols))

            return table
