/*
* Copyright (c) 2014-2017 Fredrik Mellbin
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


// TODO:
// need to remember working dir on load in case something dicks around with it
// don't write frames twice in the event that a frame really is requested twice since it's a waste of time
// have some way to make sure all frames get written? add a separate function for writing frames that isn't a filter?

#include <Magick++.h>
#include <VapourSynth.h>
#include <VSHelper.h>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <codecvt>
#include <locale>
#include <windows.h>
#else
#include <unistd.h>
#endif

// Handle both with and without hdri
#if MAGICKCORE_HDRI_ENABLE
#define IMWRI_NAMESPACE "imwrif"
#define IMWRI_PLUGIN_NAME "VapourSynth ImageMagick HDRI Writer/Reader"
#define IMWRI_ID "com.vapoursynth.imwrif"
#else
#define IMWRI_NAMESPACE "imwri"
#define IMWRI_PLUGIN_NAME "VapourSynth ImageMagick Writer/Reader"
#define IMWRI_ID "com.vapoursynth.imwri"
#endif

// Because proper namespace handling is too hard for ImageMagick shitvelopers
using MagickCore::Quantum;

//////////////////////////////////////////
// Shared

static void initMagick(VSCore *core, const VSAPI *vsapi) {
    std::string path;
#ifdef _WIN32
    const char *pathPtr = vsapi->getPluginPath(vsapi->getPluginById(IMWRI_ID, core));
    if (pathPtr) {
        path = pathPtr;
        for (auto &c : path)
            if (c == '/')
                c = '\\';
    }
#endif
    Magick::InitializeMagick(path.c_str());
}

static std::string specialPrintf(const std::string &filename, int number) {
    std::string result;
    size_t copyPos = 0;
    size_t minWidth = 0;
    bool zeroPad = false;
    bool percentSeen = false;
    bool zeroPadSeen = false;
    bool minWidthSeen = false;

    for (size_t pos = 0; pos < filename.length(); pos++) {
        const char c = filename[pos];
        if (c == '%' && !percentSeen) {
            result += filename.substr(copyPos, pos - copyPos);
            copyPos = pos;
            percentSeen = true;
            continue;
        }
        if (percentSeen) {
            if (c == '0' && !zeroPadSeen) {
                zeroPad = true;
                zeroPadSeen = true;
                continue;
            }
            if (c >= '1' && c <= '9' && !minWidthSeen) {
                minWidth = c - '0';
                zeroPadSeen = true;
                minWidthSeen = true;
                continue;
            }
            if (c == 'd') {
                std::string num = std::to_string(number);
                if (minWidthSeen && minWidth > num.length())
                    num = std::string(minWidth - num.length(), zeroPad ? '0' : ' ') + num;
                result += num;
                copyPos = pos + 1;
            }
        }
        minWidth = 0;
        zeroPad = false;
        percentSeen = false;
        zeroPadSeen = false;
        minWidthSeen = false;
    }

    result += filename.substr(copyPos, filename.length() - copyPos);

    return result;
}

static bool isAbsolute(const std::string &path) {
#ifdef _WIN32
    return path.size() > 1 && ((path[0] == '/' && path[1] == '/') || (path[0] == '\\' && path[1] == '\\') || path[1] == ':');
#else
    return path.size() && path[0] == '/';
#endif
}

static void getWorkingDir(std::string &path) {
#ifdef _WIN32
    DWORD size = GetCurrentDirectoryW(0, nullptr);

    std::vector<wchar_t> buffer(size);

    GetCurrentDirectoryW(size, buffer.data());

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> utf16;

    path = utf16.to_bytes(buffer.data()) + '\\';
#else
    char *buffer = getcwd(nullptr, 0);

    if (buffer) {
        if (buffer[0] != '(') {
            path = buffer;
            path += '/';
        }
        free(buffer);
    }
#endif
}

//////////////////////////////////////////
// Write

struct WriteData {
    VSNodeRef *videoNode;
    VSNodeRef *alphaNode;
    const VSVideoInfo *vi;
    std::string imgFormat;
    std::string filename;
    std::string workingDir;
    int firstNum;
    int quality;
    MagickCore::CompressionType compressType;
    bool dither;

    WriteData() : videoNode(nullptr), alphaNode(nullptr), vi(nullptr), quality(0), compressType(MagickCore::UndefinedCompression), dither(true) {}
};

static void VS_CC writeInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    WriteData *d = static_cast<WriteData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

template<typename T>
static void writeImageHelper(const VSFrameRef *frame, const VSFrameRef *alphaFrame, bool isGray, Magick::Image &image, int width, int height, int bitsPerSample, const VSAPI *vsapi) {
    unsigned prepeat = (MAGICKCORE_QUANTUM_DEPTH - 1) / bitsPerSample;
    unsigned pleftover = MAGICKCORE_QUANTUM_DEPTH - (bitsPerSample * prepeat);
    unsigned shiftFactor = bitsPerSample - pleftover;
    unsigned scaleFactor = 0;
    for (unsigned i = 0; i < prepeat; i++) {
        scaleFactor <<= bitsPerSample;
        scaleFactor += 1;
    }
    scaleFactor <<= pleftover;

    Magick::Pixels pixelCache(image);

    const T * VS_RESTRICT r = reinterpret_cast<const T *>(vsapi->getReadPtr(frame, 0));
    const T * VS_RESTRICT g = reinterpret_cast<const T *>(vsapi->getReadPtr(frame, isGray ? 0 : 1));
    const T * VS_RESTRICT b = reinterpret_cast<const T *>(vsapi->getReadPtr(frame, isGray ? 0 : 2));
    int strideR = vsapi->getStride(frame, 0);
    int strideG = vsapi->getStride(frame, isGray ? 0 : 1);
    int strideB = vsapi->getStride(frame, isGray ? 0 : 2);
    ssize_t rOff = pixelCache.offset(MagickCore::RedPixelChannel);
    ssize_t gOff = pixelCache.offset(MagickCore::GreenPixelChannel);
    ssize_t bOff = pixelCache.offset(MagickCore::BluePixelChannel);
    ssize_t aOff = pixelCache.offset(MagickCore::AlphaPixelChannel);
    size_t channels = image.channels();

    if (alphaFrame) {
        int strideA = vsapi->getStride(alphaFrame, 0);
        const T * VS_RESTRICT a = reinterpret_cast<const T *>(vsapi->getReadPtr(alphaFrame, 0));

        for (int y = 0; y < height; y++) {
            MagickCore::Quantum *pixels = pixelCache.get(0, y, width, 1);
            for (int x = 0; x < width; x++) {
                pixels[x * channels + rOff] = r[x] * scaleFactor + (r[x] >> shiftFactor);
                pixels[x * channels + gOff] = g[x] * scaleFactor + (g[x] >> shiftFactor);
                pixels[x * channels + bOff] = b[x] * scaleFactor + (b[x] >> shiftFactor);
                pixels[x * channels + aOff] = a[x] * scaleFactor + (a[x] >> shiftFactor);
            }

            r += strideR / sizeof(T);
            g += strideG / sizeof(T);
            b += strideB / sizeof(T);
            a += strideA / sizeof(T);

            pixelCache.sync();
        }
    } else {
        for (int y = 0; y < height; y++) {
            MagickCore::Quantum *pixels = pixelCache.get(0, y, width, 1);
            for (int x = 0; x < width; x++) {
                pixels[x * channels + rOff] = r[x] * scaleFactor + (r[x] >> shiftFactor);
                pixels[x * channels + gOff] = g[x] * scaleFactor + (g[x] >> shiftFactor);
                pixels[x * channels + bOff] = b[x] * scaleFactor + (b[x] >> shiftFactor);
                pixels[x * channels + aOff] = QuantumRange;
            }

            r += strideR / sizeof(T);
            g += strideG / sizeof(T);
            b += strideB / sizeof(T);

            pixelCache.sync();
        }
    }
}

static const VSFrameRef *VS_CC writeGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    WriteData *d = static_cast<WriteData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->videoNode, frameCtx);
        if (d->alphaNode)
            vsapi->requestFrameFilter(n, d->alphaNode, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *frame = vsapi->getFrameFilter(n, d->videoNode, frameCtx);
        const VSFormat *fi = vsapi->getFrameFormat(frame);
        int width = vsapi->getFrameWidth(frame, 0);
        int height = vsapi->getFrameHeight(frame, 0);

        const VSFrameRef *alphaFrame = nullptr;
        int alphaWidth = 0;
        int alphaHeight = 0;

        if (d->alphaNode) {
            alphaFrame = vsapi->getFrameFilter(n, d->alphaNode, frameCtx);
            alphaWidth = vsapi->getFrameWidth(alphaFrame, 0);
            alphaHeight = vsapi->getFrameHeight(alphaFrame, 0);

            if (width != alphaWidth || height != alphaHeight) {
                vsapi->setFilterError("Write: Mismatched dimension of the alpha clip", frameCtx);
                vsapi->freeFrame(frame);
                vsapi->freeFrame(alphaFrame);
                return nullptr;
            }
        }

        try {
            Magick::Image image(Magick::Geometry(width, height), Magick::Color(0, 0, 0, 0));
            image.magick(d->imgFormat);
            if (d->compressType != MagickCore::UndefinedCompression)
                image.compressType(d->compressType);
            image.quantizeDitherMethod(Magick::FloydSteinbergDitherMethod);
            image.quantizeDither(d->dither);
            image.quality(d->quality);
            if (alphaFrame)
                image.alphaChannel(Magick::ActivateAlphaChannel);

            bool isGray = fi->colorFamily == cmGray;
            if (isGray)
                image.colorSpace(Magick::GRAYColorspace);

            if (fi->bitsPerSample < static_cast<int>(image.depth()))
                image.depth(fi->bitsPerSample);

            if (fi->bytesPerSample == 4 && fi->sampleType == stFloat) {
                Magick::Pixels pixelCache(image);
                const float scaleFactor = QuantumRange;

                const float * VS_RESTRICT r = reinterpret_cast<const float *>(vsapi->getReadPtr(frame, 0));
                const float * VS_RESTRICT g = reinterpret_cast<const float *>(vsapi->getReadPtr(frame, isGray ? 0 : 1));
                const float * VS_RESTRICT b = reinterpret_cast<const float *>(vsapi->getReadPtr(frame, isGray ? 0 : 2));
               
                int strideR = vsapi->getStride(frame, 0);
                int strideG = vsapi->getStride(frame, isGray ? 0 : 1);
                int strideB = vsapi->getStride(frame, isGray ? 0 : 2);
                    
                ssize_t rOff = pixelCache.offset(MagickCore::RedPixelChannel);
                ssize_t gOff = pixelCache.offset(MagickCore::GreenPixelChannel);
                ssize_t bOff = pixelCache.offset(MagickCore::BluePixelChannel);
                ssize_t aOff = pixelCache.offset(MagickCore::AlphaPixelChannel);
                size_t channels = image.channels();

                if (alphaFrame) {
                    const float * VS_RESTRICT a = reinterpret_cast<const float *>(vsapi->getReadPtr(alphaFrame, 0));
                    int strideA = vsapi->getStride(alphaFrame, 0);
            
                    for (int y = 0; y < height; y++) {
                        MagickCore::Quantum* pixels = pixelCache.get(0, y, width, 1);
                        for (int x = 0; x < width; x++) {
                            pixels[x * channels + rOff] = r[x] * scaleFactor;
                            pixels[x * channels + gOff] = g[x] * scaleFactor;
                            pixels[x * channels + bOff] = b[x] * scaleFactor;
                            pixels[x * channels + aOff] = a[x] * scaleFactor;
                        }

                        r += strideR / sizeof(float);
                        g += strideG / sizeof(float);
                        b += strideB / sizeof(float);
                        a += strideA / sizeof(float);

                        pixelCache.sync();
                    }
                } else {
                    const float *r = reinterpret_cast<const float *>(vsapi->getReadPtr(frame, 0));
                    const float *g = reinterpret_cast<const float *>(vsapi->getReadPtr(frame, isGray ? 0 : 1));
                    const float *b = reinterpret_cast<const float *>(vsapi->getReadPtr(frame, isGray ? 0 : 2));

                    for (int y = 0; y < height; y++) {
                        MagickCore::Quantum* pixels = pixelCache.get(0, y, width, 1);
                        for (int x = 0; x < width; x++) {
                            pixels[x * channels + rOff] = r[x] * scaleFactor;
                            pixels[x * channels + gOff] = g[x] * scaleFactor;
                            pixels[x * channels + bOff] = b[x] * scaleFactor;
                            pixels[x * channels + aOff] = 1    * scaleFactor;
                        }

                        r += strideR / sizeof(float);
                        g += strideG / sizeof(float);
                        b += strideB / sizeof(float);

                        pixelCache.sync();
                    }
                }
            } else if (fi->bytesPerSample == 4) {
                writeImageHelper<uint32_t>(frame, alphaFrame, isGray, image, width, height, fi->bitsPerSample, vsapi);
            } else if (fi->bytesPerSample == 2) {
                writeImageHelper<uint16_t>(frame, alphaFrame, isGray, image, width, height, fi->bitsPerSample, vsapi);
            } else if (fi->bytesPerSample == 1) {
                writeImageHelper<uint8_t>(frame, alphaFrame, isGray, image, width, height, fi->bitsPerSample, vsapi);
            }

            std::string filename = specialPrintf(d->filename, n + d->firstNum);
            if (!isAbsolute(filename))
                filename = d->workingDir + filename;

            image.write(filename);

            vsapi->freeFrame(alphaFrame);
            return frame;
        } catch (Magick::Exception &e) {
            vsapi->setFilterError((std::string("Write: ImageMagick error: ") + e.what()).c_str(), frameCtx);
            vsapi->freeFrame(frame);
            vsapi->freeFrame(alphaFrame);
            return nullptr;
        }
    }

    return nullptr;
}

static void VS_CC writeFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    WriteData *d = static_cast<WriteData *>(instanceData);
    vsapi->freeNode(d->videoNode);
    vsapi->freeNode(d->alphaNode);
    delete d;
}

#define STR(x) #x

static void VS_CC writeCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<WriteData> d(new WriteData());
    int err = 0;

    initMagick(core, vsapi);

    d->firstNum = int64ToIntS(vsapi->propGetInt(in, "firstnum", 0, &err));
    if (d->firstNum < 0) {
        vsapi->setError(out, "Write: Frame number offset can't be negative");
        return;
    }

    d->quality = int64ToIntS(vsapi->propGetInt(in, "quality", 0, &err));
    if (err)
        d->quality = 75;
    if (d->quality < 0 || d->quality > 100) {
        vsapi->setError(out, "Write: Quality must be between 0 and 100");
        return;
    }

    const char *compressType = vsapi->propGetData(in, "compression_type", 0, &err);
    if (!err) {
        std::string s = compressType;
        std::transform(s.begin(), s.end(), s.begin(), toupper);
        if (s == "" || s == "UNDEFINED")
            d->compressType = MagickCore::UndefinedCompression;
        else if (s == "NONE")
            d->compressType = MagickCore::NoCompression;
        else if (s == "BZIP")
            d->compressType = MagickCore::BZipCompression;
        else if (s == "DXT1")
            d->compressType = MagickCore::DXT1Compression;
        else if (s == "DXT3")
            d->compressType = MagickCore::DXT3Compression;
        else if (s == "DXT5")
            d->compressType = MagickCore::DXT5Compression;
        else if (s == "FAX")
            d->compressType = MagickCore::FaxCompression;
        else if (s == "GROUP4")
            d->compressType = MagickCore::Group4Compression;
        else if (s == "JPEG")
            d->compressType = MagickCore::JPEGCompression;
        else if (s == "JPEG2000")
            d->compressType = MagickCore::JPEG2000Compression;
        else if (s == "LOSSLESSJPEG")
            d->compressType = MagickCore::LosslessJPEGCompression;
        else if (s == "LZW")
            d->compressType = MagickCore::LZWCompression;
        else if (s == "RLE")
            d->compressType = MagickCore::RLECompression;
        else if (s == "ZIP")
            d->compressType = MagickCore::ZipCompression;
        else if (s == "ZIPS")
            d->compressType = MagickCore::ZipSCompression;
        else if (s == "PIZ")
            d->compressType = MagickCore::PizCompression;
        else if (s == "PXR24")
            d->compressType = MagickCore::Pxr24Compression;
        else if (s == "B44")
            d->compressType = MagickCore::B44Compression;
        else if (s == "B44A")
            d->compressType = MagickCore::B44ACompression;
        else if (s == "LZMA")
            d->compressType = MagickCore::LZMACompression;
        else if (s == "JBIG1")
            d->compressType = MagickCore::JBIG1Compression;
        else if (s == "JBIG2")
            d->compressType = MagickCore::JBIG2Compression;
        else {
            vsapi->setError(out, "Write: Unrecognized compression type");
            return;
        }
    }

    d->videoNode = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->videoNode);
    if (!d->vi->format || (d->vi->format->colorFamily != cmRGB && d->vi->format->colorFamily != cmGray)
#if MAGICKCORE_HDRI_ENABLE
        || (d->vi->format->sampleType == stFloat && d->vi->format->bitsPerSample != 32))
#else
        || (d->vi->format->sampleType == stInteger && d->vi->format->bitsPerSample > MAGICKCORE_QUANTUM_DEPTH)
        || (d->vi->format->sampleType == stFloat))
#endif
    {
        vsapi->freeNode(d->videoNode);
#if MAGICKCORE_HDRI_ENABLE
        vsapi->setError(out, "Write: Only constant format 8-32 bit integer or float RGB and Grayscale input supported");
#else
        vsapi->setError(out, "Write: Only constant format 8-" STR(MAGICKCORE_QUANTUM_DEPTH) " bit integer RGB and Grayscale input supported");
#endif
        return;
    }

    d->alphaNode = vsapi->propGetNode(in, "alpha", 0, &err);
    d->imgFormat = vsapi->propGetData(in, "imgformat", 0, nullptr);
    d->filename = vsapi->propGetData(in, "filename", 0, nullptr);
    d->dither = !!vsapi->propGetInt(in, "dither", 0, &err);
    if (err)
        d->dither = true;

    d->vi = vsapi->getVideoInfo(d->videoNode);
    if (d->alphaNode) {
        const VSVideoInfo *alphaVi = vsapi->getVideoInfo(d->alphaNode);
        if (d->vi->width != alphaVi->width || d->vi->height != alphaVi->height || !alphaVi->format ||
            alphaVi->format != vsapi->registerFormat(cmGray, d->vi->format->sampleType, d->vi->format->bitsPerSample, 0, 0, core)) {
            vsapi->freeNode(d->videoNode);
            vsapi->freeNode(d->alphaNode);
            vsapi->setError(out, "Write: Alpha clip dimensions and format don't match the main clip");
            return;
        }
        
    }

    if (specialPrintf(d->filename, 0) == d->filename) {
        // No valid digit substitution in the filename so error out to warn the user
        vsapi->freeNode(d->videoNode);
        vsapi->freeNode(d->alphaNode);
        vsapi->setError(out, "Write: Filename string doesn't contain a number");
        return;
    }

    getWorkingDir(d->workingDir);

    vsapi->createFilter(in, out, "Write", writeInit, writeGetFrame, writeFree, fmParallelRequests, 0, d.release(), core);
}

//////////////////////////////////////////
// Read

struct ReadData {
    VSVideoInfo vi[2];
    std::vector<std::string> filenames;
    std::string workingDir;
    int firstNum;
    bool alpha;
    bool mismatch;
    bool fileListMode;
    int cachedFrameNum;
    bool cachedAlpha;
    const VSFrameRef *cachedFrame;

    ReadData() : fileListMode(true), cachedFrameNum(-1), cachedAlpha(false), cachedFrame(nullptr) {};
};

static void VS_CC readInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    ReadData *d = static_cast<ReadData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, d->alpha ? 2 : 1, node);
}

template<typename T>
static void readImageHelper(VSFrameRef *frame, VSFrameRef *alphaFrame, bool isGray, Magick::Image &image, int width, int height, int bitsPerSample, const VSAPI *vsapi) {
    unsigned shiftR = MAGICKCORE_QUANTUM_DEPTH - bitsPerSample;
    size_t channels = image.channels();
    Magick::Pixels pixelCache(image);

    T *r = reinterpret_cast<T *>(vsapi->getWritePtr(frame, 0));
    T *g = reinterpret_cast<T *>(vsapi->getWritePtr(frame, isGray ? 0 : 1));
    T *b = reinterpret_cast<T *>(vsapi->getWritePtr(frame, isGray ? 0 : 2));

    int strideR = vsapi->getStride(frame, 0);
    int strideG = vsapi->getStride(frame, isGray ? 0 : 1);
    int strideB = vsapi->getStride(frame, isGray ? 0 : 2);

    ssize_t rOff = pixelCache.offset(MagickCore::RedPixelChannel);
    ssize_t gOff = pixelCache.offset(MagickCore::GreenPixelChannel);
    ssize_t bOff = pixelCache.offset(MagickCore::BluePixelChannel);
    ssize_t aOff = pixelCache.offset(MagickCore::AlphaPixelChannel);

    if (alphaFrame) {
        T *a = reinterpret_cast<T *>(vsapi->getWritePtr(alphaFrame, 0));
        int strideA = vsapi->getStride(alphaFrame, 0);;            

        if (aOff >= 0) {
            for (int y = 0; y < height; y++) {
                const Magick::Quantum *pixels = pixelCache.getConst(0, y, width, 1);
                for (int x = 0; x < width; x++) {
                    r[x] = (unsigned)pixels[x * channels + rOff] >> shiftR;
                    g[x] = (unsigned)pixels[x * channels + gOff] >> shiftR;
                    b[x] = (unsigned)pixels[x * channels + bOff] >> shiftR;
                    a[x] = (unsigned)pixels[x * channels + aOff] >> shiftR;
                }

                r += strideR / sizeof(T);
                g += strideG / sizeof(T);
                b += strideB / sizeof(T);
                a += strideA / sizeof(T);
            }
        } else {
            memset(a, 0, strideA  * height);
        }
    } else {


        for (int y = 0; y < height; y++) {
            const Magick::Quantum *pixels = pixelCache.getConst(0, y, width, 1);
            for (int x = 0; x < width; x++) {
                r[x] = (unsigned)pixels[x * channels + rOff] >> shiftR;
                g[x] = (unsigned)pixels[x * channels + gOff] >> shiftR;
                b[x] = (unsigned)pixels[x * channels + bOff] >> shiftR;
            }

            r += strideR / sizeof(T);
            g += strideG / sizeof(T);
            b += strideB / sizeof(T);
        }
    }
}

static const VSFrameRef *VS_CC readGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    ReadData *d = static_cast<ReadData *>(*instanceData);

    if (activationReason == arInitial) {
        int index = vsapi->getOutputIndex(frameCtx);
        if (d->alpha && d->cachedFrameNum == n) {
            if ((index == 0 && !d->cachedAlpha) || (index == 1 && d->cachedAlpha)) {
                const VSFrameRef *frame = d->cachedFrame;
                d->cachedFrame = nullptr;
                d->cachedFrameNum = -1;
                return frame;
            }
        }

        VSFrameRef *frame = nullptr;
        VSFrameRef *alphaFrame = nullptr;
        
        try {
            std::string filename = d->fileListMode ? d->filenames[n] : specialPrintf(d->filenames[0], n + d->firstNum);
            if (!isAbsolute(filename))
                filename = d->workingDir + filename;

            Magick::Image image(filename);
            VSColorFamily cf = cmRGB;
            if (image.colorSpace() == Magick::GRAYColorspace)
                cf = cmGray;

            int width = static_cast<int>(image.columns());
            int height = static_cast<int>(image.rows());
            size_t channels = image.channels();

#if MAGICKCORE_HDRI_ENABLE
            VSSampleType st = stFloat;
            int depth = 32;
#else
            VSSampleType st = stInteger;
            int depth = std::min(std::max(static_cast<int>(image.depth()), 8), MAGICKCORE_QUANTUM_DEPTH);
#endif

            if (d->vi[0].format && (cf != d->vi[0].format->colorFamily || depth != d->vi[0].format->bitsPerSample)) {
                std::string err = "Read: Format mismatch for frame " + std::to_string(n) + ", is ";
                err += vsapi->registerFormat(cf, st, depth, 0, 0, core)->name + std::string(" but should be ") + d->vi[0].format->name;
                vsapi->setFilterError(err.c_str(), frameCtx);
                return nullptr;
            }

            if (d->vi[0].width && (width != d->vi[0].width || height != d->vi[0].height)) {
                std::string err = "Read: Size mismatch for frame " + std::to_string(n) + ", is " + std::to_string(width) + "x" + std::to_string(height) + " but should be " + std::to_string(d->vi[0].width) + "x" + std::to_string(d->vi[0].height);
                vsapi->setFilterError(err.c_str(), frameCtx);
                return nullptr;
            }

            frame = vsapi->newVideoFrame(d->vi[0].format ? d->vi[0].format : vsapi->registerFormat(cf, stInteger, depth, 0, 0, core), width, height, nullptr, core);
            if (d->alpha)
                alphaFrame = vsapi->newVideoFrame(d->vi[1].format ? d->vi[1].format : vsapi->registerFormat(cmGray, stInteger, depth, 0, 0, core), width, height, nullptr, core);
            const VSFormat *fi = vsapi->getFrameFormat(frame);
 
            bool isGray = fi->colorFamily == cmGray;                
     
            if (fi->bytesPerSample == 4 && fi->sampleType == stFloat) {
                const float scaleFactor = QuantumRange;
                Magick::Pixels pixelCache(image);

                float *r = reinterpret_cast<float *>(vsapi->getWritePtr(frame, 0));
                float *g = reinterpret_cast<float *>(vsapi->getWritePtr(frame, isGray ? 0 : 1));
                float *b = reinterpret_cast<float *>(vsapi->getWritePtr(frame, isGray ? 0 : 2));

                int strideR = vsapi->getStride(frame, 0);
                int strideG = vsapi->getStride(frame, isGray ? 0 : 1);
                int strideB = vsapi->getStride(frame, isGray ? 0 : 2);

                ssize_t rOff = pixelCache.offset(MagickCore::RedPixelChannel);
                ssize_t gOff = pixelCache.offset(MagickCore::GreenPixelChannel);
                ssize_t bOff = pixelCache.offset(MagickCore::BluePixelChannel);

                if (alphaFrame) {
                    float *a = reinterpret_cast<float *>(vsapi->getWritePtr(alphaFrame, 0));
                    int strideA = vsapi->getStride(alphaFrame, 0);
                    ssize_t aOff = pixelCache.offset(MagickCore::AlphaPixelChannel);

                    if (aOff >= 0) {
                        for (int y = 0; y < height; y++) {
                            const MagickCore::Quantum* pixels = pixelCache.getConst(0, y, width, 1);
                            for (int x = 0; x < width; x++) {
                                r[x] = pixels[x * channels + rOff] / scaleFactor;
                                g[x] = pixels[x * channels + gOff] / scaleFactor;
                                b[x] = pixels[x * channels + bOff] / scaleFactor;
                                a[x] = pixels[x * channels + aOff] / scaleFactor;
                            }

                            r += strideR / sizeof(float);
                            g += strideG / sizeof(float);
                            b += strideB / sizeof(float);
                            a += strideA / sizeof(float);
                        }
                    } else {
                        memset(a, 0, strideA  * height);
                    }
                } else {
                    for (int y = 0; y < height; y++) {
                        const MagickCore::Quantum* pixels = pixelCache.getConst(0, y, width, 1);
                        for (int x = 0; x < width; x++) {
                            r[x] = pixels[x * channels + rOff] / scaleFactor;
                            g[x] = pixels[x * channels + gOff] / scaleFactor;
                            b[x] = pixels[x * channels + bOff] / scaleFactor;
                        }

                        r += strideR / sizeof(float);
                        g += strideG / sizeof(float);
                        b += strideB / sizeof(float);
                    }
                }
            } else if (fi->bytesPerSample == 4) {
                readImageHelper<uint32_t>(frame, alphaFrame, isGray, image, width, height, fi->bitsPerSample, vsapi);
            } else if (fi->bytesPerSample == 2) {
                readImageHelper<uint16_t>(frame, alphaFrame, isGray, image, width, height, fi->bitsPerSample, vsapi);
            } else if (fi->bytesPerSample == 1) {
                readImageHelper<uint8_t>(frame, alphaFrame, isGray, image, width, height, fi->bitsPerSample, vsapi);
            }
        } catch (Magick::Exception &e) {
            vsapi->setFilterError((std::string("Read: ImageMagick error: ") + e.what()).c_str(), frameCtx);
            vsapi->freeFrame(frame);
            vsapi->freeFrame(alphaFrame);
            return nullptr;
        }

        if (d->alpha) {
            d->cachedFrameNum = n;
            vsapi->freeFrame(d->cachedFrame);
            if (index == 0) {
                d->cachedAlpha = true;
                d->cachedFrame = alphaFrame;
                return frame;
            } else /* if (index == 1) */ {
                d->cachedAlpha = false;
                d->cachedFrame = frame;
                return alphaFrame;
            }
        } else {
            return frame;
        }
    }

    return nullptr;
}

static void VS_CC readFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    ReadData *d = static_cast<ReadData *>(instanceData);
    delete d;
}

static void VS_CC readCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<ReadData> d(new ReadData());
    int err = 0;

    initMagick(core, vsapi);

    d->firstNum = int64ToIntS(vsapi->propGetInt(in, "firstnum", 0, &err));
    if (d->firstNum < 0) {
        vsapi->setError(out, "Read: Frame number offset can't be negative");
        return;
    }

    d->alpha = !!vsapi->propGetInt(in, "alpha", 0, &err);
    d->mismatch = !!vsapi->propGetInt(in, "mismatch", 0, &err);

    int numElem = vsapi->propNumElements(in, "filename");
    d->filenames.resize(numElem);
    for (int i = 0; i < numElem; i++)
        d->filenames[i] = vsapi->propGetData(in, "filename", i, nullptr);
    
    d->vi[0] = { nullptr, 30, 1, 0, 0, static_cast<int>(d->filenames.size()), 0 };
    // See if it's a single filename with number substitution and check how many files exist
    if (d->vi[0].numFrames == 1 && specialPrintf(d->filenames[0], 0) != d->filenames[0]) {
        d->fileListMode = false;

        for (int i = d->firstNum; i < INT_MAX; i++) {
#ifdef _WIN32
            std::string printedStr(specialPrintf(d->filenames[0], i));
            std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> conversion;
            std::wstring wPath = conversion.from_bytes(printedStr);
            FILE * f = _wfopen(wPath.c_str(), L"rb");
#else
            FILE * f = fopen(specialPrintf(d->filenames[0], i).c_str(), "rb");
#endif
            if (f) {
                fclose(f);
            } else {
                d->vi[0].numFrames = i - d->firstNum;
                break;
            }
        }

        if (d->vi[0].numFrames == 0) {
            vsapi->setError(out, "Read: No files matching the given pattern exist");
            return;
        }
    }

    try {
        Magick::Image image(d->fileListMode ? d->filenames[0] : specialPrintf(d->filenames[0], d->firstNum));
#if MAGICKCORE_HDRI_ENABLE
        VSSampleType st = stFloat;
        int depth = 32;
#else
        VSSampleType st = stInteger;
        int depth = std::min(std::max(static_cast<int>(image.depth()), 8), MAGICKCORE_QUANTUM_DEPTH);
#endif

        if (!d->mismatch || d->vi[0].numFrames == 1) {
            d->vi[0].height = static_cast<int>(image.rows());
            d->vi[0].width = static_cast<int>(image.columns());
            VSColorFamily cf = cmRGB;
            if (image.colorSpace() == Magick::GRAYColorspace)
                cf = cmGray;
            d->vi[0].format = vsapi->registerFormat(cf, st, depth, 0, 0, core);
        }

        if (d->alpha) {
            d->vi[1] = d->vi[0];
            if (d->vi[0].format)
                d->vi[1].format = vsapi->registerFormat(cmGray, d->vi[0].format->sampleType, depth, 0, 0, core);
        }
    } catch (Magick::Exception &e) {
        vsapi->setError(out, (std::string("Read: Failed to read image properties: ") + e.what()).c_str());
        return;
    }

    getWorkingDir(d->workingDir);

    vsapi->createFilter(in, out, "Read", readInit, readGetFrame, readFree, fmUnordered, 0, d.release(), core);
}


//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc(IMWRI_ID, IMWRI_NAMESPACE, IMWRI_PLUGIN_NAME, VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Write", "clip:clip;imgformat:data;filename:data;firstnum:int:opt;quality:int:opt;dither:int:opt;compression_type:data:opt;alpha:clip:opt;", writeCreate, nullptr, plugin);
    registerFunc("Read", "filename:data[];firstnum:int:opt;mismatch:int:opt;alpha:int:opt;", readCreate, nullptr, plugin);
}
