import math
import unittest

from src.nvfp4_reference import (
    FP4_POSITIVE, QuantizedTensor, decode_e2m1, decode_e4m3_scale,
    dequantize, encode_e2m1, encode_e4m3_scale, pack_e2m1, quantize,
    unpack_e2m1,
)


class FormatTests(unittest.TestCase):
    def test_all_e2m1_values_and_signs(self):
        for code, value in enumerate(FP4_POSITIVE):
            self.assertEqual(decode_e2m1(code), value)
            self.assertEqual(encode_e2m1(value), code)
            self.assertEqual(decode_e2m1(code | 8), -value)
        self.assertEqual(encode_e2m1(-0.0), 8)
        self.assertEqual(encode_e2m1(100.0), 7)
        self.assertEqual(encode_e2m1(-100.0), 15)

    def test_e2m1_midpoints_round_to_even(self):
        for x, code in zip((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0),
                           (0, 2, 2, 4, 4, 6, 6)):
            self.assertEqual(encode_e2m1(x), code)
            self.assertEqual(encode_e2m1(-x), code | 8)

    def test_e4m3_encodings_boundaries_and_ties(self):
        self.assertEqual(decode_e4m3_scale(1), 2**-9)
        self.assertEqual(decode_e4m3_scale(0x38), 1.0)
        self.assertEqual(decode_e4m3_scale(0x7e), 448.0)
        self.assertEqual(encode_e4m3_scale(1000), 0x7e)
        for code in range(126):
            a, b = decode_e4m3_scale(code), decode_e4m3_scale(code + 1)
            self.assertEqual(encode_e4m3_scale((a + b) / 2), code if code % 2 == 0 else code + 1)
            self.assertEqual(encode_e4m3_scale(a), code)

    def test_packing_order_and_padding(self):
        self.assertEqual(pack_e2m1([1, 2, 15]), bytes([0x21, 0x0f]))
        self.assertEqual(unpack_e2m1(bytes([0x21, 0x0f]), 3), [1, 2, 15])
        with self.assertRaises(ValueError):
            unpack_e2m1(bytes([0x21, 0xff]), 3)


class QuantizationTests(unittest.TestCase):
    def test_zero_empty_and_tail(self):
        self.assertEqual(dequantize(quantize([])), [])
        q = quantize([0.0] * 17)
        self.assertEqual(q.global_scale, 1.0)
        self.assertEqual(len(q.block_scales), 2)
        self.assertEqual(dequantize(q), [0.0] * 17)

    def test_independent_block_scales_and_roundtrip(self):
        values = [6.0, -6.0] * 8 + [0.375] * 16
        q = quantize(values)
        y = dequantize(q)
        self.assertNotEqual(q.block_scales[0], q.block_scales[1])
        self.assertLess(max(abs(a - b) for a, b in zip(values, y)), 0.02)
        self.assertTrue(math.isclose(q.global_scale, 6.0 / 2688, rel_tol=1e-7))

    def test_tiny_block_has_nonzero_scale(self):
        q = quantize([1.0] * 16 + [1e-12] * 16)
        self.assertEqual(q.block_scales[1], 1)
        self.assertTrue(all(math.isfinite(x) for x in dequantize(q)))

    def test_invalid_input_and_metadata(self):
        for v in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                quantize([v])
        with self.assertRaises(ValueError):
            quantize([1e-320])
        with self.assertRaises(ValueError):
            dequantize(QuantizedTensor(b'\x00', b'', 1.0, 1))


if __name__ == '__main__':
    unittest.main()
