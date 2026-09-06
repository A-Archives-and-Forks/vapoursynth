import unittest

import vapoursynth as vs


def get_pixel_value(clip, plane):
    frame = clip.get_frame(0)
    arr = frame[plane]
    return arr[0, 0]


class FilterTestSequence(unittest.TestCase):
    def setUp(self):
        self.core = vs.core
        self.Transpose = self.core.std.Transpose
        self.BlankClip = self.core.std.BlankClip
        self.MakeFullDiff = self.core.std.MakeFullDiff
        self.MergeFullDiff = self.core.std.MergeFullDiff

    def test_transpose8_test(self):
        clip = self.BlankClip(format=vs.YUV420P8, color=[0, 0, 0], width=1156, height=752)
        self.Transpose(clip).get_frame(0)

    def test_transpose16(self):
        clip = self.BlankClip(format=vs.YUV420P16, color=[0, 0, 0], width=1156, height=752)
        self.Transpose(clip).get_frame(0)

    def test_transposeS(self):
        clip = self.BlankClip(format=vs.YUV444PS, color=[0, 0, 0], width=1156, height=752)
        self.Transpose(clip).get_frame(0)

    def test_makefulldiff1(self):
        clipa = self.BlankClip(format=vs.YUV420P8, color=[0, 255, 0], width=1156, height=752)
        clipb = self.BlankClip(format=vs.YUV420P8, color=[255, 0, 0], width=1156, height=752)
        diff1 = self.MakeFullDiff(clipa, clipb)
        newclipb = self.MergeFullDiff(clipb, diff1)
        self.assertEqual(get_pixel_value(newclipb, 0), get_pixel_value(clipa, 0))
        self.assertEqual(get_pixel_value(newclipb, 1), get_pixel_value(clipa, 1))



try:
    from gputestsupport import GPUTestMixin, HAVE_GPU
except ImportError:
    from test.gputestsupport import GPUTestMixin, HAVE_GPU


@unittest.skipUnless(HAVE_GPU, "no usable Vulkan device")
class FilterTestSequenceGPU(GPUTestMixin, FilterTestSequence):
    """The same tests with every std call routed through its GPU path."""


class KernelRegressionTests(unittest.TestCase):
    """Inputs with a known scalar answer that the SIMD kernels or filter setup once got wrong."""

    def setUp(self):
        self.core = vs.core

    def test_convolution_7x7_word_maximal_coefficients(self):
        # 49 taps of 1023 on 65535 overflow a 32 bit accumulator; the normalized blur must keep the constant
        clip = self.core.std.BlankClip(format=vs.GRAY16, width=80, height=12, color=[65535], length=1)
        with self.core.std.Convolution(clip, matrix=[1023] * 49).get_frame(0) as f:
            self.assertEqual(set(v for row in f[0].tolist() for v in row), {65535})

    def test_convolution_saturates_before_converting(self):
        # value * taps / 1e-7 is far past int32; the result must saturate to the maximum, not wrap to 0
        for fmt, value, maximum in ((vs.GRAY8, 123, 255), (vs.GRAY16, 12345, 65535)):
            clip = self.core.std.BlankClip(format=fmt, width=80, height=12, color=[value], length=1)
            for size in (3, 5, 7, 9):
                with self.core.std.Convolution(clip, matrix=[1] * (size * size), divisor=1e-7).get_frame(0) as f:
                    self.assertEqual(set(v for row in f[0].tolist() for v in row), {maximum}, (fmt, size))

    def test_lut2_rejects_missing_planes(self):
        a = self.core.std.BlankClip(format=vs.YUV444P8, width=4, height=4, length=1)
        b = self.core.std.BlankClip(format=vs.GRAY8, width=4, height=4, length=1)
        with self.assertRaises(vs.Error):
            self.core.std.Lut2(a, b, lut=list(range(256)) * 256)
        # the other way round only the plane both clips have is processed
        self.core.std.Lut2(b, a, lut=list(range(256)) * 256).get_frame(0)

    def test_blankaudio_keep_partial_last_frame(self):
        clip = self.core.std.BlankAudio(length=3073, keep=True)
        for n in (1, 0, 1, 0):
            with clip.get_frame(n) as f:
                self.assertEqual(len(bytes(f[0])) // f.bytes_per_sample, 3072 if n == 0 else 1)


if __name__ == "__main__":
    unittest.main()
