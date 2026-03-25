#pragma once
#include "image.h"

float lanczos(float x, int a);
float sample_lanczos(const Image& img, float x, float y, int c, int a);