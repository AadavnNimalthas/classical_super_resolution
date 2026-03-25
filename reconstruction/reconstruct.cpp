#include "reconstruct.h"
#include "../core/upscale.h"
#include "../alignment/align.h"

#include <iostream>
#include <cmath>
#include <algorithm>

Image reconstruct(const std::vector<Image>& images, int scale)
{
    if(images.empty())
    {
        std::cerr << "No input images\n";
        std::exit(1);
    }

    const Image& ref = images[0];

    // Align
    std::vector<Shift> shifts(images.size());
    for(size_t i = 0; i < images.size(); i++)
    {
        shifts[i] = estimate_shift(ref, images[i]);
    }

    // Upscale initial guess
    Image high = upscale_lanczos(ref, scale, 3);

    int iterations = 6;
    float alpha = 0.15f;

    for(int iter = 0; iter < iterations; iter++)
    {
        std::cout << "Iteration " << iter << std::endl;

        std::vector<float> accum(high.data.size(), 0.0f);
        std::vector<int> count(high.data.size(), 0);

        for(size_t k = 0; k < images.size(); k++)
        {
            const Image& low = images[k];
            Shift shift = shifts[k];

            for(int y = 0; y < low.height; y++)
            {
                for(int x = 0; x < low.width; x++)
                {
                    float fx = (x + shift.dx) * scale;
                    float fy = (y + shift.dy) * scale;

                    int x0 = static_cast<int>(fx);
                    int y0 = static_cast<int>(fy);

                    for(int dy = 0; dy <= 1; dy++)
                    {
                        for(int dx = 0; dx <= 1; dx++)
                        {
                            int hx = x0 + dx;
                            int hy = y0 + dy;

                            if(hx < 0 || hx >= high.width || hy < 0 || hy >= high.height)
                                continue;

                            float wx = 1.0f - std::abs(fx - hx);
                            float wy = 1.0f - std::abs(fy - hy);
                            float w = wx * wy;

                            if(w <= 0.0f) continue;

                            for(int c = 0; c < low.channels; c++)
                            {
                                int idx = (hy * high.width + hx) * high.channels + c;

                                float predicted = high.data[idx];
                                float actual = low.at(x, y, c);

                                float error = actual - predicted;

                                accum[idx] += w * error;
                                count[idx]++;
                            }
                        }
                    }
                }
            }
        }

        for(size_t i = 0; i < high.data.size(); i++)
        {
            if(count[i] > 0)
            {
                high.data[i] += alpha * (accum[i] / count[i]);
            }
        }
    }

    return high;
}