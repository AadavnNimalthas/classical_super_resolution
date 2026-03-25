#include "align.h"
#include <limits>
#include <cmath>

float compute_error(const Image& ref, const Image& target, int dx, int dy)
{
    float error = 0.0f;

    int count = 0;

    for(int y = 0; y < ref.height; y++)
    {
        for(int x = 0; x < ref.width; x++)
        {
            int tx = x + dx;
            int ty = y + dy;

            if(tx < 0 || tx >= target.width || ty < 0 || ty >= target.height)
                continue;

            for(int c = 0; c < ref.channels; c++)
            {
                float diff = ref.at(x, y, c) - target.at(tx, ty, c);
                error += diff * diff;
            }

            count++;
        }
    }

    if(count == 0) return std::numeric_limits<float>::max();

    return error / count;
}

Shift estimate_shift(const Image& ref, const Image& target)
{
    int search_radius = 2;

    float best_error = std::numeric_limits<float>::max();

    int best_dx = 0;
    int best_dy = 0;

    for(int dy = -search_radius; dy <= search_radius; dy++)
    {
        for(int dx = -search_radius; dx <= search_radius; dx++)
        {
            float err = compute_error(ref, target, dx, dy);

            if(err < best_error)
            {
                best_error = err;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    Shift shift;
    shift.dx = static_cast<float>(best_dx);
    shift.dy = static_cast<float>(best_dy);

    return shift;
}