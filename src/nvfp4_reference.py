"""Readable CPU reference for one NVFP4 recipe; no CUDA or third-party packages.

Values use E2M1, each consecutive block of 16 shares an E4M3FN scale,
and a tensor shares a float32 scale: x_hat = fp4 * block_scale * global_scale.
This educational flat layout is NOT the CUTLASS scale-factor layout.
"""

from bisect import bisect_left
from dataclasses import dataclass
import math
import struct
from typing import Iterable

BLOCK_SIZE = 16
FP4_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e4m3_value(code: int) -> float:
    exponent, mantissa = divmod(code, 8)
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


# Positive E4M3FN: code 0x7f is NaN, not 480; the maximum is 448.
E4M3_POSITIVE = tuple(_e4m3_value(code) for code in range(127))


def _nearest_even(value: float, values: tuple[float, ...]) -> int:
    """Return the positive encoding nearest to value, saturating; ties to even."""
    pos = bisect_left(values, value)
    if pos == 0:
        return 0
    if pos == len(values):
        return len(values) - 1
    below, above = pos - 1, pos
    low_distance, high_distance = value - values[below], values[above] - value
    if low_distance < high_distance:
        return below
    if high_distance < low_distance:
        return above
    return below if below % 2 == 0 else above


def encode_e2m1(value: float) -> int:
    """Encode one finite value with round-to-nearest-even and saturation."""
    if not math.isfinite(value):
        raise ValueError("E2M1 input must be finite")
    magnitude = _nearest_even(abs(value), FP4_POSITIVE)
    return magnitude | (8 if math.copysign(1.0, value) < 0 else 0)


def decode_e2m1(code: int) -> float:
    if not isinstance(code, int) or not 0 <= code < 16:
        raise ValueError("E2M1 code must be an integer in [0, 15]")
    value = FP4_POSITIVE[code & 7]
    return -value if code & 8 else value


def encode_e4m3_scale(value: float) -> int:
    """Encode a nonnegative finite scale; underflow rounds to zero normally."""
    if not math.isfinite(value) or value < 0:
        raise ValueError("scale must be finite and nonnegative")
    return _nearest_even(value, E4M3_POSITIVE)


def decode_e4m3_scale(code: int) -> float:
    if not isinstance(code, int) or not 0 <= code <= 126:
        raise ValueError("scale code must be a positive E4M3FN encoding 0..126")
    return E4M3_POSITIVE[code]


def pack_e2m1(codes: Iterable[int]) -> bytes:
    """Pack consecutive codes low nibble first; pad an odd tail with zero."""
    codes = list(codes)
    if any(not isinstance(c, int) or not 0 <= c < 16 for c in codes):
        raise ValueError("all codes must be integers in [0, 15]")
    return bytes(codes[i] | ((codes[i + 1] if i + 1 < len(codes) else 0) << 4)
                 for i in range(0, len(codes), 2))


def unpack_e2m1(packed: bytes, length: int) -> list[int]:
    if not isinstance(length, int) or length < 0 or len(packed) != (length + 1) // 2:
        raise ValueError("packed byte count does not match length")
    if length % 2 and packed and packed[-1] >> 4:
        raise ValueError("odd-length padding nibble must be zero")
    return [(packed[i // 2] >> (4 * (i % 2))) & 15 for i in range(length)]


@dataclass(frozen=True)
class QuantizedTensor:
    packed: bytes
    block_scales: bytes  # Positive E4M3FN codes in sequential order.
    global_scale: float  # Rounded to IEEE float32.
    length: int


def quantize(values: Iterable[float]) -> QuantizedTensor:
    """Quantize a flat tensor. The last block may contain fewer than 16 values.

    Scale recipe: global=amax/(448*6), rounded to float32; block scale is
    round_E4M3(block_amax/(6*global)). Nonzero blocks clamp their encoded
    scale to the smallest E4M3 subnormal to avoid division by zero. Zero
    blocks use scale 1. This reference uses deterministic RNE, not stochastic
    rounding. Inputs must be finite and global scale representable in float32.
    """
    xs = [float(x) for x in values]
    if any(not math.isfinite(x) for x in xs):
        raise ValueError("all input values must be finite")
    amax = max((abs(x) for x in xs), default=0.0)
    try:
        global_scale = struct.unpack("<f", struct.pack("<f", amax / (448.0 * 6.0)))[0] if amax else 1.0
    except OverflowError as exc:
        raise ValueError("global scale is not representable in float32") from exc
    if not math.isfinite(global_scale) or global_scale == 0:
        raise ValueError("global scale is not representable as a nonzero float32")
    scales, codes = [], []
    for start in range(0, len(xs), BLOCK_SIZE):
        block = xs[start:start + BLOCK_SIZE]
        block_max = max(abs(x) for x in block)
        scale_code = max(1, encode_e4m3_scale((block_max / global_scale) / 6.0)) if block_max else encode_e4m3_scale(1.0)
        scale = decode_e4m3_scale(scale_code)
        scales.append(scale_code)
        # Divide in two steps to avoid underflow in scale * global_scale.
        codes.extend(encode_e2m1((x / global_scale) / scale) for x in block)
    return QuantizedTensor(pack_e2m1(codes), bytes(scales), global_scale, len(xs))


def dequantize(tensor: QuantizedTensor) -> list[float]:
    if len(tensor.block_scales) != (tensor.length + BLOCK_SIZE - 1) // BLOCK_SIZE:
        raise ValueError("block scale count does not match tensor length")
    if not math.isfinite(tensor.global_scale) or tensor.global_scale <= 0:
        raise ValueError("global scale must be positive and finite")
    codes = unpack_e2m1(tensor.packed, tensor.length)
    return [decode_e2m1(code) * decode_e4m3_scale(tensor.block_scales[i // BLOCK_SIZE]) * tensor.global_scale
            for i, code in enumerate(codes)]
