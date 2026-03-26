// 1.0
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include "io/image_loader.h"
#include "io/image_writer.h"
#include "reconstruction/reconstruct.h"

namespace fs = std::filesystem;

float compute_sharpness(const Image& img)
{
    float sum = 0.0f;
    float sum_sq = 0.0f;

    int pixels = img.width * img.height;

    for(int y = 0; y < img.height; y++)
    {
        for(int x = 0; x < img.width; x++)
        {
            float gray = 0.0f;

            for(int c = 0; c < img.channels; c++)
            {
                gray += img.at(x, y, c);
            }

            gray /= img.channels;

            sum += gray;
            sum_sq += gray * gray;
        }
    }

    float mean = sum / pixels;
    float variance = (sum_sq / pixels) - (mean * mean);

    return variance;
}

bool is_image_file(const fs::path& path)
{
    std::string ext = path.extension().string();

    return ext == ".png" || ext == ".jpg" || ext == ".jpeg";
}

int main()
{
    std::string folder;
    std::cout << "Enter image folder path: ";
    std::getline(std::cin, folder);

    std::vector<Image> images;

    for(const auto& entry : fs::directory_iterator(folder))
    {
        if(entry.is_regular_file() && is_image_file(entry.path()))
        {
            std::cout << "Loading: " << entry.path() << std::endl;
            images.push_back(load_image(entry.path().string()));
        }
    }

    if(images.size() < 2)
    {
        std::cerr << "Need at least 2 images\n";
        return 1;
    }

    std::cout << "Selecting best reference image...\n";

    int best_index = 0;
    float best_score = compute_sharpness(images[0]);

    for(size_t i = 1; i < images.size(); i++)
    {
        float score = compute_sharpness(images[i]);

        if(score > best_score)
        {
            best_score = score;
            best_index = i;
        }
    }

    std::cout << "Best image index: " << best_index << std::endl;

    std::swap(images[0], images[best_index]);

    int scale;
    std::cout << "Enter upscale factor (2 or 4 recommended): ";
    std::cin >> scale;

    std::cout << "Running reconstruction...\n";

    Image result = reconstruct(images, scale);

    std::string output_path = folder + "/output.png";

    save_image(output_path, result);

    std::cout << "Saved result to: " << output_path << std::endl;

    return 0;
}