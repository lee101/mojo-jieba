"""ctypes binding for the Mojo segmentation kernel."""

from __future__ import annotations

import ctypes
import os

import numpy as np
from numpy.typing import NDArray

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_JIEBA_LIB") or os.path.join(
    ROOT, "dist", "libmojo-jieba.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.isfile(LIB):
            raise RuntimeError(
                f"Mojo library not found at {LIB}; run `pixi run build` first"
            )
        _library = ctypes.CDLL(LIB)
        _library.mjb_route.argtypes = [
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            F,
            I,
            I,
        ]
        _library.mjb_route.restype = None
        _library.mjb_route_batch.argtypes = [
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            F,
            I,
            I,
        ]
        _library.mjb_route_batch.restype = None
    return _library


def _array_address(
    array: np.ndarray,
    *,
    name: str,
    dtype: np.dtype,
    writable: bool = False,
) -> int:
    """Validate an ndarray before exposing its storage to Mojo."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError(f"{name} must be C-contiguous and aligned")
    if writable and not array.flags.writeable:
        raise ValueError(f"{name} must be writable")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise ValueError(f"{name} has a null data pointer")
    return address


def route(
    chars: NDArray[np.uint32],
    edge_chars: NDArray[np.uint32],
    edge_offsets: NDArray[np.int64],
    edge_children: NDArray[np.int64],
    node_weights: NDArray[np.float64],
    root_children: NDArray[np.int64],
    max_word_length: int,
    unknown_weight: float,
    output: NDArray[np.int64],
    scores: NDArray[np.float64],
) -> None:
    """Call the single-range kernel while keeping every NumPy owner alive."""
    n = len(chars)
    _validate_common(
        chars,
        edge_chars,
        edge_offsets,
        edge_children,
        node_weights,
        root_children,
        max_word_length,
        output,
        scores,
    )
    if len(output) != n or len(scores) != n + 1:
        raise ValueError("route outputs do not match the character count")
    if not n:
        scores[0] = 0.0
        return
    lib().mjb_route(
        _array_address(chars, name="chars", dtype=np.dtype(np.uint32)),
        n,
        _array_address(edge_chars, name="edge_chars", dtype=np.dtype(np.uint32)),
        _array_address(edge_offsets, name="edge_offsets", dtype=np.dtype(np.int64)),
        _array_address(edge_children, name="edge_children", dtype=np.dtype(np.int64)),
        _array_address(node_weights, name="node_weights", dtype=np.dtype(np.float64)),
        _array_address(root_children, name="root_children", dtype=np.dtype(np.int64)),
        max_word_length,
        unknown_weight,
        _array_address(output, name="output", dtype=np.dtype(np.int64), writable=True),
        _array_address(scores, name="scores", dtype=np.dtype(np.float64), writable=True),
    )


def route_batch(
    chars: NDArray[np.uint32],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    edge_chars: NDArray[np.uint32],
    edge_offsets: NDArray[np.int64],
    edge_children: NDArray[np.int64],
    node_weights: NDArray[np.float64],
    root_children: NDArray[np.int64],
    max_word_length: int,
    unknown_weight: float,
    output: NDArray[np.int64],
    scores: NDArray[np.float64],
) -> None:
    """Validate disjoint spans and call the batch kernel."""
    _validate_common(
        chars,
        edge_chars,
        edge_offsets,
        edge_children,
        node_weights,
        root_children,
        max_word_length,
        output,
        scores,
    )
    starts_addr = _array_address(starts, name="starts", dtype=np.dtype(np.int64))
    ends_addr = _array_address(ends, name="ends", dtype=np.dtype(np.int64))
    if len(starts) != len(ends):
        raise ValueError("starts and ends must have the same length")
    if len(output) != len(chars) or len(scores) != len(chars) + 1:
        raise ValueError("route outputs do not match the character count")
    invalid = (
        np.any(starts < 0)
        or np.any(ends <= starts)
        or np.any(ends > len(chars))
        or (len(starts) > 1 and np.any(starts[1:] <= ends[:-1]))
    )
    if invalid:
        raise ValueError("route spans must be ordered, separated, and non-empty")
    if not len(starts):
        scores[0] = 0.0
        return
    lib().mjb_route_batch(
        _array_address(chars, name="chars", dtype=np.dtype(np.uint32)),
        len(chars),
        starts_addr,
        ends_addr,
        len(starts),
        _array_address(edge_chars, name="edge_chars", dtype=np.dtype(np.uint32)),
        _array_address(edge_offsets, name="edge_offsets", dtype=np.dtype(np.int64)),
        _array_address(edge_children, name="edge_children", dtype=np.dtype(np.int64)),
        _array_address(node_weights, name="node_weights", dtype=np.dtype(np.float64)),
        _array_address(root_children, name="root_children", dtype=np.dtype(np.int64)),
        max_word_length,
        unknown_weight,
        _array_address(output, name="output", dtype=np.dtype(np.int64), writable=True),
        _array_address(scores, name="scores", dtype=np.dtype(np.float64), writable=True),
    )


def _validate_common(
    chars: np.ndarray,
    edge_chars: np.ndarray,
    edge_offsets: np.ndarray,
    edge_children: np.ndarray,
    node_weights: np.ndarray,
    root_children: np.ndarray,
    max_word_length: int,
    output: np.ndarray,
    scores: np.ndarray,
) -> None:
    _array_address(chars, name="chars", dtype=np.dtype(np.uint32))
    _array_address(edge_chars, name="edge_chars", dtype=np.dtype(np.uint32))
    _array_address(edge_offsets, name="edge_offsets", dtype=np.dtype(np.int64))
    _array_address(edge_children, name="edge_children", dtype=np.dtype(np.int64))
    _array_address(node_weights, name="node_weights", dtype=np.dtype(np.float64))
    _array_address(root_children, name="root_children", dtype=np.dtype(np.int64))
    _array_address(output, name="output", dtype=np.dtype(np.int64), writable=True)
    _array_address(scores, name="scores", dtype=np.dtype(np.float64), writable=True)
    if len(edge_chars) != len(edge_children):
        raise ValueError("edge arrays must have the same length")
    if len(edge_offsets) != len(node_weights) + 1:
        raise ValueError("edge_offsets must contain one entry per node plus a sentinel")
    if len(root_children) != 65_536:
        raise ValueError("root_children must contain exactly 65,536 entries")
    if not isinstance(max_word_length, int) or max_word_length <= 0:
        raise ValueError("max_word_length must be a positive integer")
