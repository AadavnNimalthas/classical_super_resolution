// 1.0
#include "image_writer.h"
#include "stb_image_write.h"

#include <vector>
#include <iostream>

void save_image(const std::string& path, const Image& img)
{
    if(img.data.empty())
    {
        std::cerr << "Error: Image data is empty. Cannot save." << std::endl;
        return;
    }

    std::vector<unsigned char> out(img.width * img.height * img.channels);

    for(size_t i = 0; i < out.size(); i++)
    {
        float v = img.data[i];

        if(v < 0.0f) v = 0.0f;
        if(v > 1.0f) v = 1.0f;

        out[i] = static_cast<unsigned char>(v * 255.0f);
    }

    stbi_write_png(
        path.c_str(),
        img.width,
        img.height,
        img.channels,
        out.data(),
        img.width * img.channels
    );
}