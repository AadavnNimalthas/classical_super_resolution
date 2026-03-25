#pragma once
#include "../core/image.h"

struct Shift
{
    float dx;
    float dy;
};

Shift estimate_shift(const Image& ref, const Image& target);