import struct
import numpy as np


def readIdxInt(file):
    data = file.read(4)
    return struct.unpack(">i", data)[0]


def readImages(path):
    with open(path, "rb") as f:
        mn = readIdxInt(f)
        if mn != 2051:
            print("File did not contain expected magic number")
            return None

        numEntries = readIdxInt(f)
        imgWidth = readIdxInt(f)
        imgHeight = readIdxInt(f)

        if imgWidth <= 0 or imgHeight <= 0 or numEntries <= 0:
            return None

        vectorDim = imgWidth * imgHeight
        result = np.empty((numEntries, vectorDim), dtype=np.float32)

        for e in xrange(numEntries):
            for i in xrange(vectorDim):
                pixelBuf = f.read(1)
                result[e, i] = struct.unpack("B", pixelBuf)[0] / 255.0

        return result


def readLabels(path):
    with open(path, "rb") as f:
        mn = readIdxInt(f)
        if mn != 2049:
            print("File did not contain expected magic number")
            return None

        numEntries = readIdxInt(f)

        result = np.empty((numEntries, 10), dtype=np.float32)
        result.fill(0.0)

        for e in xrange(numEntries):
            digitBuf = f.read(1)
            digit = struct.unpack("B", digitBuf)[0]
            assert(digit >= 0 and digit <= 9)

            result[e][digit] = 1.0

        return result
