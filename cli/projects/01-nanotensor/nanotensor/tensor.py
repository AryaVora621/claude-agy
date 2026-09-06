"""
NanoTensor: Pure Python zero-dependency autograd tensor engine.
Features reverse-mode automatic differentiation, multi-dimensional broadcasting,
contiguous memory flattening with strided indexing, and full gradient accumulation.
"""

from __future__ import annotations
import math
import random
from typing import Sequence, Union, Tuple, Optional, List, Callable


def _compute_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculates row-major contiguous strides for a given shape."""
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _broadcast_shapes(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
    """Determines standard NumPy-style broadcasted shape between two shapes."""
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)
    pad1 = (1,) * (max_len - len1) + shape1
    pad2 = (1,) * (max_len - len2) + shape2
    result = []
    for d1, d2 in zip(pad1, pad2):
        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise ValueError(f"Cannot broadcast shapes {shape1} and {shape2}")
    return tuple(result)


def _reduce_broadcast_grad(grad: Tensor, target_shape: Tuple[int, ...]) -> Tensor:
    """
    Sums gradient along axes where broadcasting expanded dimensions,
    restoring gradient shape to match the target parameter shape.
    """
    if grad.shape == target_shape:
        return grad

    grad_dim = len(grad.shape)
    target_dim = len(target_shape)
    padded_target = (1,) * (grad_dim - target_dim) + target_shape

    current = grad
    # Sum out leading dimensions that were prepended during broadcasting
    for i in range(grad_dim - target_dim):
        current = current.sum(axis=0, keepdim=False)

    # Sum out dimensions that had size 1 and were expanded
    for i, (gd, td) in enumerate(zip(current.shape, target_shape)):
        if td == 1 and gd > 1:
            current = current.sum(axis=i, keepdim=True)

    if current.shape != target_shape:
        current = current.reshape(target_shape)
    return current


class Tensor:
    """
    Multidimensional array with reverse-mode automatic differentiation.
    Backed by a flat 1D Python list with strided coordinate mapping.
    """

    def __init__(
        self,
        data: Union[float, int, Sequence, List],
        shape: Optional[Tuple[int, ...]] = None,
        strides: Optional[Tuple[int, ...]] = None,
        offset: int = 0,
        requires_grad: bool = False,
        _children: Tuple[Tensor, ...] = (),
        _op: str = "",
    ):
        if isinstance(data, (int, float)):
            self._data = [float(data)]
            self.shape = () if shape is None else tuple(shape)
        elif isinstance(data, list) and shape is not None:
            self._data = [float(x) for x in data]
            self.shape = tuple(shape)
        else:
            flat_list, inferred_shape = self._flatten(data)
            self._data = flat_list
            self.shape = inferred_shape if shape is None else tuple(shape)

        self.strides = _compute_strides(self.shape) if strides is None else tuple(strides)
        self.offset = offset
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op

    @classmethod
    def _flatten(cls, nested: Sequence) -> Tuple[List[float], Tuple[int, ...]]:
        """Recursively flattens arbitrary nested lists into a 1D list and extracts shape."""
        if not isinstance(nested, (list, tuple)):
            return [float(nested)], ()
        if len(nested) == 0:
            return [], (0,)

        shape = [len(nested)]
        first = nested[0]
        if isinstance(first, (list, tuple)):
            sub_flat, sub_shape = cls._flatten(first)
            shape.extend(sub_shape)
            total_flat = []
            for item in nested:
                flat_item, _ = cls._flatten(item)
                total_flat.extend(flat_item)
            return total_flat, tuple(shape)
        else:
            return [float(x) for x in nested], tuple(shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        count = 1
        for d in self.shape:
            count *= d
        return count

    def tolist(self) -> Union[float, List]:
        """Converts tensor back into nested Python list representation."""
        if self.ndim == 0:
            return self._data[self.offset]

        def _build_nested(indices: List[int], dim: int):
            if dim == self.ndim - 1:
                row = []
                for i in range(self.shape[dim]):
                    flat_idx = self._get_flat_index(indices + [i])
                    row.append(self._data[flat_idx])
                return row
            nested = []
            for i in range(self.shape[dim]):
                nested.append(_build_nested(indices + [i], dim + 1))
            return nested

        return _build_nested([], 0)

    def _get_flat_index(self, multi_idx: Sequence[int]) -> int:
        """Translates multi-dimensional coordinates to flat buffer index using strides."""
        idx = self.offset
        for i, s in zip(multi_idx, self.strides):
            idx += i * s
        return idx

    def contiguous(self) -> Tensor:
        """Returns a contiguous copy of the tensor buffer if currently strided/non-contiguous."""
        expected_strides = _compute_strides(self.shape)
        if self.strides == expected_strides and self.offset == 0 and len(self._data) == self.size:
            return self

        new_data = [0.0] * self.size
        # Iterate over all elements in row-major order
        if self.size == 0:
            return Tensor([], self.shape, requires_grad=self.requires_grad)

        def _copy_recursive(indices: List[int], dim: int, dest_idx: int) -> int:
            if dim == self.ndim:
                new_data[dest_idx] = self._data[self._get_flat_index(indices)]
                return dest_idx + 1
            for i in range(self.shape[dim]):
                indices.append(i)
                dest_idx = _copy_recursive(indices, dim + 1, dest_idx)
                indices.pop()
            return dest_idx

        _copy_recursive([], 0, 0)
        return Tensor(new_data, self.shape, requires_grad=self.requires_grad)

    def __repr__(self) -> str:
        arr_str = str(self.tolist())
        grad_str = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"Tensor({arr_str}, shape={self.shape}{grad_str})"

    # -------------------------------------------------------------------------
    # Factories
    # -------------------------------------------------------------------------

    @classmethod
    def zeros(cls, shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
        count = 1
        for d in shape:
            count *= d
        return cls([0.0] * count, shape=shape, requires_grad=requires_grad)

    @classmethod
    def ones(cls, shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
        count = 1
        for d in shape:
            count *= d
        return cls([1.0] * count, shape=shape, requires_grad=requires_grad)

    @classmethod
    def randn(cls, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, requires_grad: bool = False) -> Tensor:
        count = 1
        for d in shape:
            count *= d
        data = [random.gauss(mean, std) for _ in range(count)]
        return cls(data, shape=shape, requires_grad=requires_grad)

    @classmethod
    def uniform(cls, shape: Tuple[int, ...], low: float = -1.0, high: float = 1.0, requires_grad: bool = False) -> Tensor:
        count = 1
        for d in shape:
            count *= d
        data = [random.uniform(low, high) for _ in range(count)]
        return cls(data, shape=shape, requires_grad=requires_grad)

    # -------------------------------------------------------------------------
    # Structural Operations (Reshape, Transpose, Permute, Squeeze, Slice)
    # -------------------------------------------------------------------------

    def reshape(self, new_shape: Tuple[int, ...]) -> Tensor:
        """Reshapes tensor into a new shape having the identical total element count."""
        # Handle -1 infer dimension
        total_elements = self.size
        computed_shape = list(new_shape)
        neg_idx = -1
        known_prod = 1
        for i, d in enumerate(new_shape):
            if d == -1:
                if neg_idx != -1:
                    raise ValueError("Only one dimension can be -1 in reshape")
                neg_idx = i
            else:
                known_prod *= d

        if neg_idx != -1:
            if known_prod == 0 or total_elements % known_prod != 0:
                raise ValueError(f"Cannot infer dimension -1 for shape {new_shape} from size {total_elements}")
            computed_shape[neg_idx] = total_elements // known_prod
        final_shape = tuple(computed_shape)

        new_count = 1
        for d in final_shape:
            new_count *= d
        if new_count != total_elements:
            raise ValueError(f"Cannot reshape tensor of size {total_elements} into {final_shape}")

        # Ensure buffer is contiguous before altering shape
        c = self.contiguous()
        out = Tensor(c._data, shape=final_shape, requires_grad=self.requires_grad, _children=(self,), _op="reshape")

        def _backward():
            if self.requires_grad:
                grad_self = out.grad.reshape(self.shape)
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad = self.grad + grad_self

        out._backward = _backward
        return out

    def transpose(self, dim0: int = -2, dim1: int = -1) -> Tensor:
        """Swaps two dimensions of the tensor."""
        dims = list(range(self.ndim))
        # Handle negative dimension indexing
        d0 = dim0 if dim0 >= 0 else self.ndim + dim0
        d1 = dim1 if dim1 >= 0 else self.ndim + dim1
        dims[d0], dims[d1] = dims[d1], dims[d0]
        return self.permute(tuple(dims))

    def permute(self, dims: Tuple[int, ...]) -> Tensor:
        """Permutes dimensions according to specified permutation order."""
        if len(dims) != self.ndim:
            raise ValueError(f"Permute dims {dims} must match tensor ndim {self.ndim}")
        norm_dims = [d if d >= 0 else self.ndim + d for d in dims]
        new_shape = tuple(self.shape[d] for d in norm_dims)
        new_strides = tuple(self.strides[d] for d in norm_dims)

        out = Tensor(self._data, shape=new_shape, strides=new_strides, offset=self.offset,
                     requires_grad=self.requires_grad, _children=(self,), _op="permute")

        # Inverse permutation for backpropagation
        inv_dims = [0] * self.ndim
        for i, d in enumerate(norm_dims):
            inv_dims[d] = i
        inv_dims_tuple = tuple(inv_dims)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad.permute(inv_dims_tuple).contiguous()
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad = self.grad + grad_self

        out._backward = _backward
        return out

    def unsqueeze(self, dim: int) -> Tensor:
        """Inserts a singleton dimension of size 1 at specified position."""
        d = dim if dim >= 0 else self.ndim + dim + 1
        new_shape = self.shape[:d] + (1,) + self.shape[d:]
        return self.reshape(new_shape)

    def squeeze(self, dim: Optional[int] = None) -> Tensor:
        """Removes singleton dimensions of size 1."""
        if dim is None:
            new_shape = tuple(d for d in self.shape if d != 1)
        else:
            d = dim if dim >= 0 else self.ndim + dim
            if self.shape[d] != 1:
                return self
            new_shape = self.shape[:d] + self.shape[d + 1:]
        return self.reshape(new_shape)

    # -------------------------------------------------------------------------
    # Core Math & Autograd Operations
    # -------------------------------------------------------------------------

    def __add__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        broadcast_shape = _broadcast_shapes(self.shape, other.shape)
        a_contig = self.contiguous()
        b_contig = other.contiguous()

        # Allocate contiguous output buffer
        total_size = 1
        for d in broadcast_shape:
            total_size *= d
        out_data = [0.0] * total_size

        # Compute flat coordinates using modulo / stride arithmetic
        out_strides = _compute_strides(broadcast_shape)

        for out_idx in range(total_size):
            # Unravel index into multidimensional coordinates
            rem = out_idx
            coords = []
            for s in out_strides:
                coords.append(rem // s)
                rem %= s

            # Map to operand flat indices with broadcasting
            idx_a = 0
            pad_a = len(broadcast_shape) - len(self.shape)
            for i, d in enumerate(self.shape):
                coord = coords[i + pad_a] if d > 1 else 0
                idx_a += coord * a_contig.strides[i]

            idx_b = 0
            pad_b = len(broadcast_shape) - len(other.shape)
            for i, d in enumerate(other.shape):
                coord = coords[i + pad_b] if d > 1 else 0
                idx_b += coord * b_contig.strides[i]

            out_data[out_idx] = a_contig._data[idx_a] + b_contig._data[idx_b]

        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(out_data, shape=broadcast_shape, requires_grad=req_grad, _children=(self, other), _op="+")

        def _backward():
            if self.requires_grad:
                grad_a = _reduce_broadcast_grad(out.grad, self.shape)
                self.grad = grad_a if self.grad is None else self.grad + grad_a
            if other.requires_grad:
                grad_b = _reduce_broadcast_grad(out.grad, other.shape)
                other.grad = grad_b if other.grad is None else other.grad + grad_b

        out._backward = _backward
        return out

    def __radd__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self.__add__(other)

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other + (-self)

    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        broadcast_shape = _broadcast_shapes(self.shape, other.shape)
        a_contig = self.contiguous()
        b_contig = other.contiguous()

        total_size = 1
        for d in broadcast_shape:
            total_size *= d
        out_data = [0.0] * total_size
        out_strides = _compute_strides(broadcast_shape)

        for out_idx in range(total_size):
            rem = out_idx
            coords = []
            for s in out_strides:
                coords.append(rem // s)
                rem %= s

            idx_a = 0
            pad_a = len(broadcast_shape) - len(self.shape)
            for i, d in enumerate(self.shape):
                coord = coords[i + pad_a] if d > 1 else 0
                idx_a += coord * a_contig.strides[i]

            idx_b = 0
            pad_b = len(broadcast_shape) - len(other.shape)
            for i, d in enumerate(other.shape):
                coord = coords[i + pad_b] if d > 1 else 0
                idx_b += coord * b_contig.strides[i]

            out_data[out_idx] = a_contig._data[idx_a] * b_contig._data[idx_b]

        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(out_data, shape=broadcast_shape, requires_grad=req_grad, _children=(self, other), _op="*")

        def _backward():
            if self.requires_grad:
                grad_a = _reduce_broadcast_grad(out.grad * other, self.shape)
                self.grad = grad_a if self.grad is None else self.grad + grad_a
            if other.requires_grad:
                grad_b = _reduce_broadcast_grad(out.grad * self, other.shape)
                other.grad = grad_b if other.grad is None else other.grad + grad_b

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1.0)

    def __rtruediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * (self ** -1.0)

    def __pow__(self, power: float) -> Tensor:
        c = self.contiguous()
        out_data = [x ** power for x in c._data]
        out = Tensor(out_data, shape=self.shape, requires_grad=self.requires_grad, _children=(self,), _op=f"**{power}")

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * (power * (c ** (power - 1.0)))
                self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Nonlinear Activations
    # -------------------------------------------------------------------------

    def relu(self) -> Tensor:
        c = self.contiguous()
        out_data = [x if x > 0.0 else 0.0 for x in c._data]
        out = Tensor(out_data, shape=self.shape, requires_grad=self.requires_grad, _children=(self,), _op="ReLU")

        def _backward():
            if self.requires_grad:
                grad_data = [g if x > 0.0 else 0.0 for x, g in zip(c._data, out.grad.contiguous()._data)]
                grad_self = Tensor(grad_data, shape=self.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        c = self.contiguous()
        # Clamped to avoid float overflow
        out_data = []
        for x in c._data:
            if x >= 0:
                z = math.exp(-x)
                out_data.append(1.0 / (1.0 + z))
            else:
                z = math.exp(x)
                out_data.append(z / (1.0 + z))

        out = Tensor(out_data, shape=self.shape, requires_grad=self.requires_grad, _children=(self,), _op="Sigmoid")

        def _backward():
            if self.requires_grad:
                # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                grad_data = [g * s * (1.0 - s) for s, g in zip(out_data, out.grad.contiguous()._data)]
                grad_self = Tensor(grad_data, shape=self.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def silu(self) -> Tensor:
        """
        SiLU (Swish) activation function: x * sigmoid(x).
        Standard activation unit in modern LLMs (e.g., LLaMA, Mistral, Qwen SwiGLU).
        Derivative: s + x * s * (1 - s) = s * (1 + x * (1 - s)).
        """
        c = self.contiguous()
        sig_data = []
        for x in c._data:
            if x >= 0:
                z = math.exp(-x)
                sig_data.append(1.0 / (1.0 + z))
            else:
                z = math.exp(x)
                sig_data.append(z / (1.0 + z))

        out_data = [x * s for x, s in zip(c._data, sig_data)]
        out = Tensor(out_data, shape=self.shape, requires_grad=self.requires_grad, _children=(self,), _op="SiLU")

        def _backward():
            if self.requires_grad:
                grad_data = [g * s * (1.0 + x * (1.0 - s))
                             for x, s, g in zip(c._data, sig_data, out.grad.contiguous()._data)]
                grad_self = Tensor(grad_data, shape=self.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        c = self.contiguous()
        out_data = [math.exp(min(max(x, -80.0), 80.0)) for x in c._data]
        out = Tensor(out_data, shape=self.shape, requires_grad=self.requires_grad, _children=(self,), _op="exp")

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * out
                self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        c = self.contiguous()
        out_data = [math.log(max(x, 1e-15)) for x in c._data]
        out = Tensor(out_data, shape=self.shape, requires_grad=self.requires_grad, _children=(self,), _op="log")

        def _backward():
            if self.requires_grad:
                grad_self = out.grad / self
                self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Reductions: Sum, Mean
    # -------------------------------------------------------------------------

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> Tensor:
        """Sums elements over given axis or axes."""
        if axis is None:
            # Full reduction to scalar
            c = self.contiguous()
            total = sum(c._data)
            out_shape = (1,) * self.ndim if keepdim else ()
            out = Tensor(total, shape=out_shape, requires_grad=self.requires_grad, _children=(self,), _op="sum")

            def _backward():
                if self.requires_grad:
                    grad_val = out.grad._data[0]
                    grad_self = Tensor([grad_val] * self.size, shape=self.shape)
                    self.grad = grad_self if self.grad is None else self.grad + grad_self

            out._backward = _backward
            return out

        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        axes = tuple(a if a >= 0 else self.ndim + a for a in axes)

        # Compute output shape
        out_shape_list = []
        for i, d in enumerate(self.shape):
            if i in axes:
                if keepdim:
                    out_shape_list.append(1)
            else:
                out_shape_list.append(d)
        out_shape = tuple(out_shape_list)

        c = self.contiguous()
        out_size = 1
        for d in out_shape:
            out_size *= d
        out_data = [0.0] * out_size

        # Accumulate sums
        for in_idx in range(self.size):
            rem = in_idx
            coords = []
            for s in c.strides:
                coords.append(rem // s)
                rem %= s

            # Determine destination index in output
            out_coords = []
            for i, coord in enumerate(coords):
                if i in axes:
                    if keepdim:
                        out_coords.append(0)
                else:
                    out_coords.append(coord)

            out_strides = _compute_strides(out_shape)
            out_idx = sum(co * st for co, st in zip(out_coords, out_strides)) if out_coords else 0
            out_data[out_idx] += c._data[in_idx]

        out = Tensor(out_data, shape=out_shape, requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                # Reshape out.grad to keepdim shape so it broadcasts back cleanly
                expanded_shape = list(self.shape)
                for a in axes:
                    expanded_shape[a] = 1
                reshaped_grad = out.grad.reshape(tuple(expanded_shape))

                # Repeat / expand back to self.shape
                grad_self = Tensor.zeros(self.shape) + reshaped_grad
                self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> Tensor:
        """Computes arithmetic mean over specified axis or all elements."""
        s = self.sum(axis=axis, keepdim=keepdim)
        count = self.size / s.size
        return s * (1.0 / count)

    # -------------------------------------------------------------------------
    # Matrix Multiplication
    # -------------------------------------------------------------------------

    def matmul(self, other: Tensor) -> Tensor:
        """
        Batched matrix multiplication: (..., M, K) @ (..., K, N) -> (..., M, N).
        Supports arbitrary leading batch dimensions with standard broadcasting.
        """
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError(f"Both tensors must be at least 2D for matmul, got {self.shape} and {other.shape}")

        *batch_a, M, Ka = self.shape
        *batch_b, Kb, N = other.shape

        if Ka != Kb:
            raise ValueError(f"Inner matrix dimensions must match: {Ka} vs {Kb} in {self.shape} @ {other.shape}")

        batch_out = _broadcast_shapes(tuple(batch_a), tuple(batch_b))
        out_shape = batch_out + (M, N)

        a_contig = self.contiguous()
        b_contig = other.contiguous()

        batch_count = 1
        for d in batch_out:
            batch_count *= d

        out_size = batch_count * M * N
        out_data = [0.0] * out_size

        batch_strides_out = _compute_strides(batch_out) if batch_out else ()

        for b_idx in range(batch_count):
            # Unpack batch coordinates
            rem = b_idx
            coords = []
            for s in batch_strides_out:
                coords.append(rem // s)
                rem %= s

            # Coordinate offset in A
            offset_a = 0
            pad_a = len(batch_out) - len(batch_a)
            for i, d in enumerate(batch_a):
                co = coords[i + pad_a] if d > 1 else 0
                offset_a += co * a_contig.strides[i]

            # Coordinate offset in B
            offset_b = 0
            pad_b = len(batch_out) - len(batch_b)
            for i, d in enumerate(batch_b):
                co = coords[i + pad_b] if d > 1 else 0
                offset_b += co * b_contig.strides[i]

            # 2D GEMM for this batch slice
            stride_a_row = a_contig.strides[-2]
            stride_a_col = a_contig.strides[-1]
            stride_b_row = b_contig.strides[-2]
            stride_b_col = b_contig.strides[-1]

            offset_out = b_idx * (M * N)

            for m in range(M):
                row_a_idx = offset_a + m * stride_a_row
                row_out_idx = offset_out + m * N
                for n in range(N):
                    col_b_idx = offset_b + n * stride_b_col
                    acc = 0.0
                    for k in range(Ka):
                        acc += a_contig._data[row_a_idx + k * stride_a_col] * b_contig._data[col_b_idx + k * stride_b_row]
                    out_data[row_out_idx + n] = acc

        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(out_data, shape=out_shape, requires_grad=req_grad, _children=(self, other), _op="@")

        def _backward():
            if self.requires_grad:
                # grad_a = out.grad @ other.T
                b_t = other.transpose(-1, -2)
                grad_a = out.grad.matmul(b_t)
                grad_a_reduced = _reduce_broadcast_grad(grad_a, self.shape)
                self.grad = grad_a_reduced if self.grad is None else self.grad + grad_a_reduced

            if other.requires_grad:
                # grad_b = self.T @ out.grad
                a_t = self.transpose(-1, -2)
                grad_b = a_t.matmul(out.grad)
                grad_b_reduced = _reduce_broadcast_grad(grad_b, other.shape)
                other.grad = grad_b_reduced if other.grad is None else other.grad + grad_b_reduced

        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)

    # -------------------------------------------------------------------------
    # Softmax & Cross Entropy Loss
    # -------------------------------------------------------------------------

    def softmax(self, axis: int = -1) -> Tensor:
        """
        Numerically stable softmax along specified axis:
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        """
        ax = axis if axis >= 0 else self.ndim + axis
        c = self.contiguous()

        # Find maximum along axis for numerical stability
        stride_ax = c.strides[ax]
        dim_len = self.shape[ax]

        # Calculate prefix strides
        outer_count = 1
        for i in range(ax):
            outer_count *= self.shape[i]
        inner_count = 1
        for i in range(ax + 1, self.ndim):
            inner_count *= self.shape[i]

        out_data = [0.0] * self.size

        for o in range(outer_count):
            o_offset = o * (dim_len * inner_count)
            for inn in range(inner_count):
                base_idx = o_offset + inn
                # Find max
                max_val = -float("inf")
                for k in range(dim_len):
                    val = c._data[base_idx + k * inner_count]
                    if val > max_val:
                        max_val = val

                # Compute exp sum
                exp_sum = 0.0
                for k in range(dim_len):
                    e = math.exp(c._data[base_idx + k * inner_count] - max_val)
                    out_data[base_idx + k * inner_count] = e
                    exp_sum += e

                # Normalize
                inv_sum = 1.0 / exp_sum
                for k in range(dim_len):
                    out_data[base_idx + k * inner_count] *= inv_sum

        out = Tensor(out_data, shape=self.shape, requires_grad=self.requires_grad, _children=(self,), _op="softmax")

        def _backward():
            if self.requires_grad:
                # Analytical gradient of softmax:
                # dL/dx_i = y_i * (dL/dy_i - sum_k(dL/dy_k * y_k))
                grad_contig = out.grad.contiguous()
                grad_self_data = [0.0] * self.size

                for o in range(outer_count):
                    o_offset = o * (dim_len * inner_count)
                    for inn in range(inner_count):
                        base_idx = o_offset + inn
                        # Dot product of grad and softmax output
                        sum_gy = 0.0
                        for k in range(dim_len):
                            idx = base_idx + k * inner_count
                            sum_gy += grad_contig._data[idx] * out_data[idx]

                        for k in range(dim_len):
                            idx = base_idx + k * inner_count
                            y_k = out_data[idx]
                            g_k = grad_contig._data[idx]
                            grad_self_data[idx] = y_k * (g_k - sum_gy)

                grad_self = Tensor(grad_self_data, shape=self.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Backpropagation Engine
    # -------------------------------------------------------------------------

    def backward(self, gradient: Optional[Tensor] = None) -> None:
        """
        Executes reverse-mode automatic differentiation starting from this tensor.
        Performs topological sorting of DAG nodes to propagate gradients cleanly.
        """
        if self.grad is None:
            if gradient is None:
                if self.size != 1:
                    raise RuntimeError("Grad can only be implicitly created for scalar outputs")
                self.grad = Tensor.ones(self.shape)
            else:
                self.grad = gradient

        # Build topological sort of graph
        topo: List[Tensor] = []
        visited = set()

        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Call backward on each node in reverse topological sequence
        for node in reversed(topo):
            node._backward()
