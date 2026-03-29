// 1.1
#pragma once
#include "../core/image.h"

struct Shift
{
    float dx    = 0.0f;
    float dy    = 0.0f;
    float theta = 0.0f;   // rotation in radians, around image centre
};

Shift estimate_shift(const Image& ref, const Image& target);