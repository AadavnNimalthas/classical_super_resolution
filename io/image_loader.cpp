#include "image_loader.h"
#include "stb_image.h"

#include <iostream>
#include <cstdlib>

Image load_image(const std::string& path)
{
    int w = 0;
    int h = 0;
    int c = 0;

    int desired_channels = 3;

    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &c, desired_channels);

    if(raw == nullptr)
    {
        std::cerr << "Failed to load image: " << path << std::endl;
        std::exit(1);
    }

    Image img;
    img.width = w;
    img.height = h;
    img.channels = desired_channels;

    int total = w * h * desired_channels;
    img.data.resize(total);

    for(int i = 0; i < total; i++)
    {
        img.data[i] = static_cast<float>(raw[i]) / 255.0f;
    }

    stbi_image_free(raw);

    return img;
}