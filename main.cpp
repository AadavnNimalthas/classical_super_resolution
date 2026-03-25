#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include "io/image_loader.h"
#include "io/image_writer.h"
#include "reconstruction/reconstruct.h"
#include "utils/system_info.h"

namespace fs = std::filesystem;

bool is_supported_image_file(const std::string& path)
{
    std::string ext = fs::path(path).extension().string();

    for(char& ch : ext)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    if(ext == ".png")
    {
        return true;
    }
    else if(ext == ".jpg")
    {
        return true;
    }
    else if(ext == ".jpeg")
    {
        return true;
    }
    else if(ext == ".bmp")
    {
        return true;
    }
    else if(ext == ".tga")
    {
        return true;
    }

    return false;
}

int main()
{
    std::cout << "Classical Super Resolution Pipeline\n";

    std::string folder_path;
    std::cout << "Enter image folder path (use . for current folder): ";
    std::getline(std::cin, folder_path);

    if(folder_path.empty())
    {
        folder_path = ".";
    }

    if(!fs::exists(folder_path) || !fs::is_directory(folder_path))
    {
        std::cout << "Invalid folder path\n";
        return 1;
    }

    std::vector<std::string> file_paths;

    for(const auto& entry : fs::directory_iterator(folder_path))
    {
        if(entry.is_regular_file())
        {
            std::string path = entry.path().string();

            if(is_supported_image_file(path))
            {
                file_paths.push_back(path);
            }
        }
    }

    if(file_paths.empty())
    {
        std::cout << "No supported image files found in folder\n";
        return 1;
    }

    std::vector<Image> images;

    for(const std::string& path : file_paths)
    {
        images.push_back(load_image(path));
    }

    if(images.empty())
    {
        std::cout << "No images loaded\n";
        return 1;
    }

    std::cout << "Loaded " << images.size() << " image(s)\n";

    int requested_scale = 2;
    std::cout << "Enter scale factor (2 or 4): ";
    std::cin >> requested_scale;

    if(requested_scale != 2 && requested_scale != 4)
    {
        requested_scale = 2;
    }

    PerformanceTier tier = SystemInfo::getPerformanceTier();
    int final_scale = requested_scale;

    if(tier == PerformanceTier::LOW)
    {
        final_scale = 2;
        std::cout << "Low tier system detected. Using 2x.\n";
    }
    else if(tier == PerformanceTier::MEDIUM)
    {
        if(final_scale > 2)
        {
            final_scale = 2;
        }
        std::cout << "Medium tier system detected.\n";
    }
    else
    {
        std::cout << "High tier system detected.\n";
    }

    Image result = reconstruct(images, final_scale);

    std::string output_path;
    std::cout << "Enter output file name (example: output.png): ";
    std::cin >> output_path;

    if(output_path.empty())
    {
        output_path = "output.png";
    }

    save_image(output_path, result);

    std::cout << "Done. Saved to " << output_path << "\n";

    return 0;
}