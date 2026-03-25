#pragma once
#include <vector>

struct Image
{
    int width;
    int height;
    int channels;
    std::vector<float> data;

    Image() : width(0), height(0), channels(0) {}

    Image(int w, int h, int c)
    {
        width = w;
        height = h;
        channels = c;
        data.resize(w * h * c);
    }

    float& at(int x, int y, int c)
    {
        return data[(y * width + x) * channels + c];
    }

    const float& at(int x, int y, int c) const
    {
        return data[(y * width + x) * channels + c];
    }
};