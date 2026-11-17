"""Run from the project root: python examples/demo_quantization.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nvfp4_reference import (  # noqa: E402
    BLOCK_SIZE, decode_e4m3_scale, dequantize, quantize, unpack_e2m1,
)


def main():
    # Two blocks with very different magnitudes illustrate per-block scaling.
    values = [0.0, 0.25, -0.5, 1.0, -1.5, 2.0, -3.0, 6.0] * 2
    values += [v / 10 for v in values]
    quantized = quantize(values)
    restored = dequantize(quantized)
    codes = unpack_e2m1(quantized.packed, quantized.length)

    print(f"global scale (FP32): {quantized.global_scale:.9g}")
    for block in range(len(quantized.block_scales)):
        code = quantized.block_scales[block]
        print(f"block {block}: E4M3 scale code=0x{code:02x}, "
              f"value={decode_e4m3_scale(code):.9g}")
    print("\nindex  block    input    code   restored     abs error")
    for i, (original, code, result) in enumerate(zip(values, codes, restored)):
        print(f"{i:5}  {i // BLOCK_SIZE:5}  {original:8.4f}    0x{code:x}  "
              f"{result:9.4f}  {abs(original - result):12.6f}")
    print(f"\nPacked {len(values)} FP4 values into {len(quantized.packed)} bytes; "
          f"plus {len(quantized.block_scales)} scale bytes and one FP32 global scale.")


if __name__ == "__main__":
    main()
