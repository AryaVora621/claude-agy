"""
SynapseDB: Vectorized Columnar Types, Bitmaps, and Data Chunks.
Provides zero-dependency Arrow-like columnar memory buffers,
null validity bitmasks, selection vectors, and RecordBatch chunks.
"""

from enum import Enum, auto
import array
from typing import List, Dict, Any, Optional, Sequence


class DataType(Enum):
    INT64 = auto()
    FLOAT64 = auto()
    VARCHAR = auto()
    BOOLEAN = auto()
    TIMESTAMP = auto()

    def __str__(self) -> str:
        return self.name


class SelectionVector:
    """
    Dense array of row indices representing rows that satisfy filter predicates.
    Enables zero-copy vectorized filtering across columnar chunks.
    """
    __slots__ = ("indices", "count")

    def __init__(self, indices: Optional[Sequence[int]] = None):
        if indices is not None:
            self.indices = array.array("I", indices)
            self.count = len(self.indices)
        else:
            self.indices = array.array("I")
            self.count = 0

    def append(self, idx: int) -> None:
        self.indices.append(idx)
        self.count += 1

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, i: int) -> int:
        return self.indices[i]

    def __iter__(self):
        return iter(self.indices)


class NullBitmap:
    """
    Compact bitmask tracking NULL validity.
    Bit 1 represents a valid non-null value, bit 0 represents NULL.
    """
    __slots__ = ("bytes_data", "size")

    def __init__(self, size: int, all_valid: bool = True):
        self.size = size
        num_bytes = (size + 7) // 8
        init_val = 0xFF if all_valid else 0x00
        self.bytes_data = bytearray([init_val] * num_bytes)

    def is_valid(self, idx: int) -> bool:
        if idx >= self.size:
            raise IndexError("Index out of bounds")
        byte_idx = idx // 8
        bit_idx = idx % 8
        return bool(self.bytes_data[byte_idx] & (1 << bit_idx))

    def set_valid(self, idx: int, valid: bool) -> None:
        if idx >= self.size:
            raise IndexError("Index out of bounds")
        byte_idx = idx // 8
        bit_idx = idx % 8
        if valid:
            self.bytes_data[byte_idx] |= (1 << bit_idx)
        else:
            self.bytes_data[byte_idx] &= ~(1 << bit_idx)


class Vector:
    """
    Single contiguous typed columnar array.
    Supports fast vectorized primitive storage (int/float/bool) via native arrays
    and variable-length strings with null bitmasks.
    """
    __slots__ = ("data_type", "data", "null_bitmap", "length")

    def __init__(
        self,
        data_type: DataType,
        data: Sequence[Any],
        nulls: Optional[Sequence[bool]] = None
    ):
        self.data_type = data_type
        self.length = len(data)

        if data_type == DataType.INT64:
            self.data = array.array("q", [0 if v is None else int(v) for v in data])
        elif data_type == DataType.FLOAT64:
            self.data = array.array("d", [0.0 if v is None else float(v) for v in data])
        elif data_type == DataType.BOOLEAN:
            self.data = array.array("B", [0 if v is None else (1 if v else 0) for v in data])
        elif data_type in (DataType.VARCHAR, DataType.TIMESTAMP):
            self.data = ["" if v is None else str(v) for v in data]
        else:
            self.data = list(data)

        self.null_bitmap = NullBitmap(self.length, all_valid=True)
        if nulls is not None:
            for idx, is_null in enumerate(nulls):
                if is_null:
                    self.null_bitmap.set_valid(idx, False)
        else:
            for idx, v in enumerate(data):
                if v is None:
                    self.null_bitmap.set_valid(idx, False)

    def is_null(self, idx: int) -> bool:
        return not self.null_bitmap.is_valid(idx)

    def get(self, idx: int) -> Any:
        if self.is_null(idx):
            return None
        val = self.data[idx]
        if self.data_type == DataType.BOOLEAN:
            return bool(val)
        return val

    def to_list(self) -> List[Any]:
        return [self.get(i) for i in range(self.length)]

    def filter(self, sel: SelectionVector) -> "Vector":
        """Produce a filtered sub-vector using selection vector row indices."""
        sub_data = [self.get(idx) for idx in sel]
        return Vector(self.data_type, sub_data)

    def slice(self, offset: int, count: int) -> "Vector":
        """Produce a contiguous slice of the vector."""
        sub_data = [self.get(i) for i in range(offset, min(self.length, offset + count))]
        return Vector(self.data_type, sub_data)

    def __len__(self) -> int:
        return self.length


class RecordBatch:
    """
    Columnar chunk containing a set of equal-length named Vectors.
    Unit of transfer through the vectorized query engine.
    """
    __slots__ = ("columns", "names", "num_rows")

    def __init__(self, columns: Dict[str, Vector]):
        self.columns = columns
        self.names = list(columns.keys())
        if self.columns:
            first_col = next(iter(columns.values()))
            self.num_rows = len(first_col)
            for name, col in columns.items():
                if len(col) != self.num_rows:
                    raise ValueError(f"Column length mismatch: {name} has {len(col)}, expected {self.num_rows}")
        else:
            self.num_rows = 0

    def column(self, name: str) -> Vector:
        if name not in self.columns:
            raise KeyError(f"Column '{name}' not in RecordBatch. Available: {self.names}")
        return self.columns[name]

    def filter(self, sel: SelectionVector) -> "RecordBatch":
        """Apply selection vector across all columns simultaneously."""
        new_cols = {name: col.filter(sel) for name, col in self.columns.items()}
        return RecordBatch(new_cols)

    def slice(self, offset: int, count: int) -> "RecordBatch":
        new_cols = {name: col.slice(offset, count) for name, col in self.columns.items()}
        return RecordBatch(new_cols)

    def project(self, column_names: Sequence[str]) -> "RecordBatch":
        """Extract only requested columns (projection pruning)."""
        new_cols = {name: self.columns[name] for name in column_names}
        return RecordBatch(new_cols)

    def to_rows(self) -> List[Dict[str, Any]]:
        """Convert columnar format into list of row dictionaries."""
        rows = []
        for i in range(self.num_rows):
            row = {name: self.columns[name].get(i) for name in self.names}
            rows.append(row)
        return rows

    def __len__(self) -> int:
        return self.num_rows
