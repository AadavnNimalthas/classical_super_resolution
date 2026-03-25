#include "reconstruct.h"
#include "../core/upscale.h"
#include "../alignment/align.h"

#include <iostream>
#include <cmath>

Image reconstruct(const std::vector<Image>& images, int scale)
{
    if(images.empty())
    {
        std::cerr << "No input images\n";
        exit(1);
    }

    // Step 1: reference image
    const Image& ref = images[0];

    // Step 2: align all images to reference
    std::vector<Shift> shifts;

    for(size_t i = 0; i < images.size(); i++)
    {
        Shift s = estimate_shift(ref, images[i]);
        shifts.push_back(s);
    }

    // Step 3: upscale reference
    Image high = upscale_lanczos(ref, scale, 3);

    int iterations = 5;

    // Step 4: iterative refinement
    for(int iter = 0; iter < iterations; iter++)
    {
        std::cout << "Iteration " << iter << std::endl;

        // For each image
        for(size_t k = 0; k < images.size(); k++)
        {
            const Image& low = images[k];
            Shift shift = shifts[k];

            // Compare low-res and simulated low-res
            for(int y = 0; y < low.height; y++)
            {
                for(int x = 0; x < low.width; x++)
                {
                    int hx = static_cast<int>((x + shift.dx) * scale);
                    int hy = static_cast<int>((y + shift.dy) * scale);

                    if(hx < 0 || hx >= high.width || hy < 0 || hy >= high.height)
                        continue;

                    for(int c = 0; c < low.channels; c++)
                    {
                        float predicted = high.at(hx, hy, c);
                        float actual = low.at(x, y, c);

                        float error = actual - predicted;

                        // back-project error
                        high.at(hx, hy, c) += 0.25f * error;
                    }
                }
            }
        }
    }

    return high;
}