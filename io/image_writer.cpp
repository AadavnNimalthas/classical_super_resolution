#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image_writer.h"
#include <vector>

void save_image(const std::string& path, const Image& img)
{
    std::vector<unsigned char> out(img.width * img.height * img.channels);

    for(int i = 0; i < out.size(); i++)
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