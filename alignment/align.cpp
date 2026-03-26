#include "align.h"
#include <limits>
#include <cmath>
#include <algorithm>
#include <omp.h>

float compute_error(const Image& ref, const Image& target, int dx, int dy)
{
    if(ref.channels != target.channels)
        return std::numeric_limits<float>::max();

    float error = 0.0f;
    int count = 0;

    int channels = ref.channels;

    // Parallelize pixel comparison
    #pragma omp parallel
    {
        float local_error = 0.0f;
        int local_count = 0;

        #pragma omp for collapse(2)
        for(int y = 0; y < ref.height; y++)
        {
            for(int x = 0; x < ref.width; x++)
            {
                int tx = x + dx;
                int ty = y + dy;

                if(tx < 0 || tx >= target.width || ty < 0 || ty >= target.height)
                    continue;

                for(int c = 0; c < channels; c++)
                {
                    float diff = ref.at(x, y, c) - target.at(tx, ty, c);
                    local_error += diff * diff;
                    local_count++;
                }
            }
        }

        #pragma omp atomic
        error += local_error;

        #pragma omp atomic
        count += local_count;
    }

    if(count == 0)
        return std::numeric_limits<float>::max();

    return error / count;
}

Shift estimate_shift(const Image& ref, const Image& target)
{
    const int search_radius = 15;

    float best_error = std::numeric_limits<float>::max();
    int best_dx = 0;
    int best_dy = 0;

    #pragma omp parallel
    {
        float local_best_error = std::numeric_limits<float>::max();
        int local_best_dx = 0;
        int local_best_dy = 0;

        #pragma omp for collapse(2)
        for(int dy = -search_radius; dy <= search_radius; dy++)
        {
            for(int dx = -search_radius; dx <= search_radius; dx++)
            {
                float err = compute_error(ref, target, dx, dy);

                if(err < local_best_error)
                {
                    local_best_error = err;
                    local_best_dx = dx;
                    local_best_dy = dy;
                }
            }
        }

        #pragma omp critical
        {
            if(local_best_error < best_error)
            {
                best_error = local_best_error;
                best_dx = local_best_dx;
                best_dy = local_best_dy;
            }
        }
    }

    Shift shift;
    shift.dx = static_cast<float>(best_dx);
    shift.dy = static_cast<float>(best_dy);

    return shift;
}