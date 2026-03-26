// 1.0
#include "align.h"
#include <limits>
#include <cmath>
#include <omp.h>

// -------------------- Bilinear Sampling --------------------
float bilinear(const Image& img, float x, float y, int c)
{
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if(x0 < 0 || x1 >= img.width || y0 < 0 || y1 >= img.height)
        return 0.0f;

    float dx = x - x0;
    float dy = y - y0;

    float v00 = img.at(x0, y0, c);
    float v10 = img.at(x1, y0, c);
    float v01 = img.at(x0, y1, c);
    float v11 = img.at(x1, y1, c);

    float v0 = v00 * (1.0f - dx) + v10 * dx;
    float v1 = v01 * (1.0f - dx) + v11 * dx;

    return v0 * (1.0f - dy) + v1 * dy;
}

// -------------------- Error Function (subpixel) --------------------
float compute_error(const Image& ref, const Image& target, float dx, float dy)
{
    if(ref.channels != target.channels)
        return std::numeric_limits<float>::max();

    float error = 0.0f;
    int count = 0;

    int channels = ref.channels;

    #pragma omp parallel
    {
        float local_error = 0.0f;
        int local_count = 0;

        #pragma omp for collapse(2)
        for(int y = 0; y < ref.height; y++)
        {
            for(int x = 0; x < ref.width; x++)
            {
                float tx = x + dx;
                float ty = y + dy;

                if(tx < 0 || tx >= target.width - 1 || ty < 0 || ty >= target.height - 1)
                    continue;

                for(int c = 0; c < channels; c++)
                {
                    float tval = bilinear(target, tx, ty, c);
                    float diff = ref.at(x, y, c) - tval;
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

// -------------------- Downscale (2x) --------------------
Image downscale_half(const Image& img)
{
    Image out;
    out.width = img.width / 2;
    out.height = img.height / 2;
    out.channels = img.channels;
    out.data.resize(out.width * out.height * out.channels);

    for(int y = 0; y < out.height; y++)
    {
        for(int x = 0; x < out.width; x++)
        {
            for(int c = 0; c < out.channels; c++)
            {
                float sum = 0.0f;

                for(int dy = 0; dy < 2; dy++)
                {
                    for(int dx = 0; dx < 2; dx++)
                    {
                        sum += img.at(x * 2 + dx, y * 2 + dy, c);
                    }
                }

                out.at(x, y, c) = sum * 0.25f;
            }
        }
    }

    return out;
}

// -------------------- Main Alignment --------------------
Shift estimate_shift(const Image& ref, const Image& target)
{
    // ---------- Step 1: Downscale ----------
    Image ref_small = downscale_half(ref);
    Image tgt_small = downscale_half(target);

    // ---------- Step 2: Coarse Search ----------
    int coarse_radius = 50;

    float best_error = std::numeric_limits<float>::max();
    int best_dx = 0;
    int best_dy = 0;

    #pragma omp parallel
    {
        float local_best_error = std::numeric_limits<float>::max();
        int local_best_dx = 0;
        int local_best_dy = 0;

        #pragma omp for collapse(2)
        for(int dy = -coarse_radius; dy <= coarse_radius; dy++)
        {
            for(int dx = -coarse_radius; dx <= coarse_radius; dx++)
            {
                float err = compute_error(ref_small, tgt_small, (float)dx, (float)dy);

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

    // ---------- Step 3: Scale back ----------
    float base_dx = best_dx * 2.0f;
    float base_dy = best_dy * 2.0f;

    // ---------- Step 4: Fine Search ----------
    float refined_dx = base_dx;
    float refined_dy = base_dy;
    float refined_error = std::numeric_limits<float>::max();

    for(float dy = base_dy - 3.0f; dy <= base_dy + 3.0f; dy += 0.5f)
    {
        for(float dx = base_dx - 3.0f; dx <= base_dx + 3.0f; dx += 0.5f)
        {
            float err = compute_error(ref, target, dx, dy);

            if(err < refined_error)
            {
                refined_error = err;
                refined_dx = dx;
                refined_dy = dy;
            }
        }
    }

    // ---------- Step 5: Subpixel Refinement ----------
    float final_dx = refined_dx;
    float final_dy = refined_dy;
    float final_error = refined_error;

    for(float dy = refined_dy - 1.0f; dy <= refined_dy + 1.0f; dy += 0.25f)
    {
        for(float dx = refined_dx - 1.0f; dx <= refined_dx + 1.0f; dx += 0.25f)
        {
            float err = compute_error(ref, target, dx, dy);

            if(err < final_error)
            {
                final_error = err;
                final_dx = dx;
                final_dy = dy;
            }
        }
    }

    Shift shift;
    shift.dx = final_dx;
    shift.dy = final_dy;

    return shift;
}