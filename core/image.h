// 1.0
#pragma once
#include <vector>
#include <cassert>

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
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        assert(c >= 0 && c < channels);

        return data[(y * width + x) * channels + c];
    }

    const float& at(int x, int y, int c) const
    {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        assert(c >= 0 && c < channels);

        return data[(y * width + x) * channels + c];
    }
};