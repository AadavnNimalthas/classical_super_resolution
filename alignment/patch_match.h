// 1.1
#pragma once
#include <vector>
#include "align.h"   // Shift is defined here — do not redefine it

struct Patch
{
    int x;
    int y;
    int size;
};

struct IntShift
{
    int dx;
    int dy;
};

Patch selectPatch(const std::vector<float>& img, int width, int height, int patchSize);

IntShift estimateShiftPatch(
    const std::vector<float>& ref,
    const std::vector<float>& tgt,
    int width,
    int height,
    Patch patch,
    int radius
);