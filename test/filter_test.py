import os
import subprocess
import sys
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


# Run in a fresh interpreter with VS_VULKAN_FORCE_STAGING set: the staging rings only exist on
# the staging paths, which that switch selects everywhere, and it is read when the device is
# created. Prints grown, shrunk, released, recreated and the unified memory flag.
STAGING_PROBE = '''
import time
import vapoursynth as vs

core = vs.core


def used():
    # staging lands in the host pool on a discrete card and in the VRAM pool on unified memory
    return core.used_cache_size + core.vulkan_device_info["allocated"]


def transfer(width, height, count):
    # every node is dropped before the caller measures, so only what outlives nodes counts:
    # committed blocks, which never change here, and the staging slots
    clip = core.std.BlankClip(format=vs.GRAY8, width=width, height=height, length=count)
    for n in range(count):
        with core.std.GPUDownload(core.std.GPUUpload(clip)).get_frame(n):
            pass


transfer(256, 128, 1)  # commits the pooled block and one 1 MiB slot per ring
base = used()
transfer(4096, 2048, 1)  # 8 MiB planes: the next slot of each ring is created at 8 MiB
grown = used() - base
transfer(256, 128, 139)  # past two demand epochs of small frames: every slot is back at 1 MiB
shrunk = used() - base

# a GPU frame kept alive pins its block, so trimming cannot move the numbers below
keep = core.std.GPUUpload(core.std.BlankClip(format=vs.GRAY8, width=64, height=64, length=1)).get_frame(0)
core.max_cache_size = 1  # 1 MB: the accounted staging alone puts the host pool over its limit
time.sleep(1.2)  # a ring must have been idle for a second before a pressure sweep releases it
released = None
clip = core.std.BlankClip(format=vs.GRAY8, width=64, height=64, length=1000)
deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    for n in range(20):  # completed calls sample the pressure, sweeps run 150 ms apart under it
        with clip.get_frame(n):
            pass
    if used() < base:
        released = used() - base
        break
    time.sleep(0.05)
transfer(256, 128, 1)  # the slots come back on demand
recreated = used() - base
print(grown, shrunk, "none" if released is None else released, recreated, int(core.vulkan_device_info["unified_memory"]))
'''


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

    @unittest.skipUnless(HAVE_GPU, "no Vulkan device")
    def test_gpu_resize_frame_matrix_outranks_matrix_in(self):
        # a special matrix_in kind is a fallback for untagged frames on the GPU path as on the CPU one
        src = self.core.std.BlankClip(format=vs.YUV444PS, width=8, height=8, color=[0.5, 0.1, 0.2], length=1)
        src = self.core.std.SetFrameProps(src, _Matrix=1, _Transfer=1, _Primaries=1)
        for matrix in ("2020cl", "ictcp"):
            cpu = self.core.resize.Point(src, format=vs.RGBS, matrix_in_s=matrix)
            gpu = self.core.std.GPUDownload(self.core.resize.Point(self.core.std.GPUUpload(src), format=vs.RGBS, matrix_in_s=matrix))
            with cpu.get_frame(0) as a, gpu.get_frame(0) as b:
                for p in range(3):
                    self.assertAlmostEqual(a[p][0, 0], b[p][0, 0], places=5, msg=matrix)

    @unittest.skipUnless(HAVE_GPU, "no Vulkan device")
    def test_transfer_staging_accounted_sized_to_demand_and_released_when_idle(self):
        env = dict(os.environ, VS_VULKAN_FORCE_STAGING="1")
        result = subprocess.run([sys.executable, "-c", STAGING_PROBE], env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)
        grown, shrunk, released, recreated, unified = result.stdout.split()
        MiB = 1 << 20
        # one 8 MiB frame each way creates one 8 MiB slot per ring next to the 1 MiB one
        self.assertGreaterEqual(int(grown), 16 * MiB, result.stdout)
        # two epochs of small frames later the big slots are gone; four 1 MiB slots per ring at most
        self.assertLessEqual(int(shrunk), 8 * MiB, result.stdout)
        # pressure only reaches the host pool here; on unified memory the staging is VRAM
        if unified == "0":
            self.assertNotEqual(released, "none", "idle staging was not released under pressure: " + result.stdout)
            self.assertLess(int(released), 0, result.stdout)
            self.assertGreaterEqual(int(recreated) - int(released), 2 * MiB, result.stdout)


if __name__ == "__main__":
    unittest.main()
