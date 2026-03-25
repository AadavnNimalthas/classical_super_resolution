#pragma once
#include <string>
#include <vector>

struct Image {
    int width;
    int height;
    int channels;
    std::vector<float> data;
};

Image load_image(const std::string& path);