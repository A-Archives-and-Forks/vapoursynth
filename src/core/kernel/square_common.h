/*
* Copyright (c) 2026 Fredrik Mellbin
*
* This file is part of VapourSynth.
*
* VapourSynth is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* VapourSynth is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with VapourSynth; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/

/* Shared between the x86 and NEON square convolution kernels: decisions about a matrix that
   every backend has to make the same way. */

#ifndef SQUARE_COMMON_H
#define SQUARE_COMMON_H

#include <cstdint>

// Whether an N*N word convolution can run in the int32 path: the biased tap products reach
// 32768 * |m| whatever the depth, and the debiased sum reaches maxval * |m|, so the larger of
// the two times the coefficient magnitudes has to stay below 2^31. Always true up to 5x5 with
// the +-1023 coefficient limit, never at 9x9 and up, and a property of the matrix at 7x7,
// where 49 taps of 1023 on 16 bit samples reach 3.3e9 and wrapped the interior to black.
template <unsigned N>
static bool sq_word_fits_i32(const int16_t *m, unsigned maxval)
{
    int64_t sum = 0;
    for (unsigned i = 0; i < N * N; ++i)
        sum += m[i] < 0 ? -static_cast<int64_t>(m[i]) : static_cast<int64_t>(m[i]);
    return sum * static_cast<int64_t>(maxval < 32768 ? 32768 : maxval) <= INT32_MAX;
}

#endif
