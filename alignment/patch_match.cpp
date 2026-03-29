#include "patch_match.h"
#include <cmath>
#include <limits>

static inline float get_pixel(
    const std::vector<float>& img,
    int width,
    int height,
    int x,
    int y
)
{
    if (x < 0 || x >= width || y < 0 || y >= height)
    {
        return 0.0f;
    }

    return img[y * width + x];
}

Patch selectPatch(
    const std::vector<float>& img,
    int width,
    int height,
    int patchSize
)
{
    int best_x = width / 2;
    int best_y = height / 2;

    float best_score = -1.0f;

    for (int y = patchSize; y < height - patchSize; y += patchSize)
    {
        for (int x = patchSize; x < width - patchSize; x += patchSize)
        {
            float score = 0.0f;

            for (int j = -patchSize / 2; j < patchSize / 2; j++)
            {
                for (int i = -patchSize / 2; i < patchSize / 2; i++)
                {
                    float gx =
                        get_pixel(img, width, height, x + i + 1, y + j) -
                        get_pixel(img, width, height, x + i - 1, y + j);

                    float gy =
                        get_pixel(img, width, height, x + i, y + j + 1) -
                        get_pixel(img, width, height, x + i, y + j - 1);

                    score += std::fabs(gx) + std::fabs(gy);
                }
            }

            if (score > best_score)
            {
                best_score = score;
                best_x = x;
                best_y = y;
            }
        }
    }

    Patch p;
    p.x = best_x;
    p.y = best_y;
    p.size = patchSize;
    return p;
}

static float patch_error(
    const std::vector<float>& ref,
    const std::vector<float>& tgt,
    int width,
    int height,
    Patch patch,
    int dx,
    int dy
)
{
    float error = 0.0f;
    int count = 0;

    for (int j = -patch.size / 2; j < patch.size / 2; j++)
    {
        for (int i = -patch.size / 2; i < patch.size / 2; i++)
        {
            int rx = patch.x + i;
            int ry = patch.y + j;

            int tx = rx + dx;
            int ty = ry + dy;

            if (rx < 0 || rx >= width || ry < 0 || ry >= height)
            {
                continue;
            }

            if (tx < 0 || tx >= width || ty < 0 || ty >= height)
            {
                continue;
            }

            float diff = ref[ry * width + rx] - tgt[ty * width + tx];
            error += diff * diff;
            count++;
        }
    }

    if (count > 0)
    {
        return error / count;
    }
    else
    {
        return std::numeric_limits<float>::max();
    }
}

IntShift estimateShiftPatch(
    const std::vector<float>& ref,
    const std::vector<float>& tgt,
    int width,
    int height,
    Patch patch,
    int radius
)
{
    IntShift best;
    best.dx = 0;
    best.dy = 0;

    float best_error = std::numeric_limits<float>::max();

    for (int dy = -radius; dy <= radius; dy++)
    {
        for (int dx = -radius; dx <= radius; dx++)
        {
            float err = patch_error(ref, tgt, width, height, patch, dx, dy);

            if (err < best_error)
            {
                best_error = err;
                best.dx = dx;
                best.dy = dy;
            }
        }
    }

    return best;
}