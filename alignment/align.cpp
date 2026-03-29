// 1.1
#include "align.h"
#include "patch_match.h"
#include <vector>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <omp.h>

// ─────────────────────────────────────────────────────────────────────────────
// Bilinear sample (unchanged from original)
// ─────────────────────────────────────────────────────────────────────────────
float bilinear(const Image& img, float x, float y, int c)
{
    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if (x0 < 0 || x1 >= img.width || y0 < 0 || y1 >= img.height)
        return 0.0f;

    float dx = x - x0;
    float dy = y - y0;

    float v00 = img.at(x0, y0, c);
    float v10 = img.at(x1, y0, c);
    float v01 = img.at(x0, y1, c);
    float v11 = img.at(x1, y1, c);

    float v0 = v00 * (1 - dx) + v10 * dx;
    float v1 = v01 * (1 - dx) + v11 * dx;

    return v0 * (1 - dy) + v1 * dy;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_error — gradient-domain error with rotation (theta) support.
//
//   For each reference pixel (x, y) the corresponding target coordinate is:
//     tx = cos(theta)*(x-cx) - sin(theta)*(y-cy) + cx + dx
//     ty = sin(theta)*(x-cx) + cos(theta)*(y-cy) + cy + dy
//
//   Bilinear interpolation is used for the rotated target lookup.
//   Gradient magnitude gate (< 1e-6) skips flat regions.
// ─────────────────────────────────────────────────────────────────────────────
float compute_error(const Image& ref, const Image& target,
                    float dx, float dy, float theta)
{
    float error = 0.0f;
    int   count = 0;

    const float cx   = ref.width  * 0.5f;
    const float cy   = ref.height * 0.5f;
    const float cosT = std::cos(theta);
    const float sinT = std::sin(theta);

    #pragma omp parallel for reduction(+:error,count) collapse(2)
    for (int y = 1; y < ref.height - 1; y += 2)
    {
        for (int x = 1; x < ref.width - 1; x += 2)
        {
            float rx = x - cx;
            float ry = y - cy;
            float tx = cosT * rx - sinT * ry + cx + dx;
            float ty = sinT * rx + cosT * ry + cy + dy;

            // Ensure that both tx-1 and tx+1 are inside the image
            if (tx < 1.0f || ty < 1.0f ||
                tx >= target.width  - 2.0f ||
                ty >= target.height - 2.0f)
                continue;

            for (int c = 0; c < ref.channels; c++)
            {
                float rgx = ref.at(x + 1, y, c) - ref.at(x - 1, y, c);
                float rgy = ref.at(x, y + 1, c) - ref.at(x, y - 1, c);

                float mag = rgx * rgx + rgy * rgy;
                if (mag < 1e-6f) continue;

                float tgx = bilinear(target, tx + 1, ty,     c)
                          - bilinear(target, tx - 1, ty,     c);
                float tgy = bilinear(target, tx,     ty + 1, c)
                          - bilinear(target, tx,     ty - 1, c);

                float dxg = rgx - tgx;
                float dyg = rgy - tgy;

                error += dxg * dxg + dyg * dyg;
                count++;
            }
        }
    }

    return count > 0 ? error / count : std::numeric_limits<float>::max();
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: 2× box-filter downscale with proper handling of odd dimensions.
// ─────────────────────────────────────────────────────────────────────────────
static void downscale_half(
    const std::vector<float>& src, int sw, int sh,
    std::vector<float>& dst, int& dw, int& dh)
{
    dw = (sw + 1) / 2;
    dh = (sh + 1) / 2;
    dst.assign(dw * dh, 0.0f);

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < dh; y++)
    {
        for (int x = 0; x < dw; x++)
        {
            int sx = x * 2;
            int sy = y * 2;
            float sum = src[sy * sw + sx];
            int cnt = 1;
            if (sx + 1 < sw) { sum += src[sy * sw + sx + 1]; cnt++; }
            if (sy + 1 < sh) { sum += src[(sy + 1) * sw + sx]; cnt++; }
            if (sx + 1 < sw && sy + 1 < sh) { sum += src[(sy + 1) * sw + sx + 1]; cnt++; }
            dst[y * dw + x] = sum / cnt;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: pick the highest-gradient patch inside a rectangular sub-region.
// ─────────────────────────────────────────────────────────────────────────────
static Patch selectPatchInRegion(
    const std::vector<float>& img, int width, int height,
    int rx0, int ry0, int rx1, int ry1, int patchSize)
{
    int half = patchSize / 2;
    int step = 1;   // check every pixel for best gradient

    float bestScore = -1.0f;
    Patch best{rx0, ry0, patchSize};

    for (int y = ry0 + half; y < ry1 - half; y += step)
    {
        for (int x = rx0 + half; x < rx1 - half; x += step)
        {
            float score = 0.0f;
            for (int j = -half; j < half; j++)
            {
                for (int i = -half; i < half; i++)
                {
                    int px = x + i;
                    int py = y + j;
                    if (px <= 0 || px >= width - 1 || py <= 0 || py >= height - 1)
                        continue;
                    float gx = img[py * width + (px + 1)] - img[py * width + (px - 1)];
                    float gy = img[(py + 1) * width + px] - img[(py - 1) * width + px];
                    score += gx * gx + gy * gy;
                }
            }
            if (score > bestScore)
            {
                bestScore = score;
                best = {x - half, y - half, patchSize};
            }
        }
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: select n patches distributed across the image in a grid layout.
//   Each grid cell yields its own best-gradient patch → spatially diverse set.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<Patch> selectDistributedPatches(
    const std::vector<float>& img, int width, int height,
    int patchSize, int n)
{
    int gridW = (int)std::ceil(std::sqrt((float)n));
    int gridH = (n + gridW - 1) / gridW;

    int cellW = width  / gridW;
    int cellH = height / gridH;

    std::vector<Patch> patches;
    patches.reserve(n);

    for (int gy = 0; gy < gridH && (int)patches.size() < n; gy++)
    {
        for (int gx = 0; gx < gridW && (int)patches.size() < n; gx++)
        {
            int x0 = gx * cellW;
            int y0 = gy * cellH;
            int x1 = std::min(x0 + cellW, width);
            int y1 = std::min(y0 + cellH, height);

            if (x1 - x0 < patchSize || y1 - y0 < patchSize)
                continue;

            patches.push_back(
                selectPatchInRegion(img, width, height,
                                    x0, y0, x1, y1, patchSize));
        }
    }
    return patches;
}

// ─────────────────────────────────────────────────────────────────────────────
// estimate_shift
//
//  Pipeline:
//    1. Grayscale
//    2. Downscale → 50%
//    3. Multi-patch coarse search at half-res (9 distributed patches, median)
//    4. Coarse rotation + translation refinement at half-res
//    5. Subpixel refinement at full resolution
//
//  Returns Shift{ dx, dy, theta }.
//  Signature unchanged; theta field is new and backward-compatible.
// ─────────────────────────────────────────────────────────────────────────────
Shift estimate_shift(const Image& ref, const Image& target)
{
    // ── 1. Grayscale ──────────────────────────────────────────────────────────
    std::vector<float> ref_gray(ref.width * ref.height);
    std::vector<float> tgt_gray(target.width * target.height);

    #pragma omp parallel for
    for (int y = 0; y < ref.height; y++)
    {
        for (int x = 0; x < ref.width; x++)
        {
            float r = 0, t = 0;
            for (int c = 0; c < ref.channels; c++)
            {
                r += ref.at(x, y, c);
                t += target.at(x, y, c);
            }
            ref_gray[y * ref.width  + x] = r / ref.channels;
            tgt_gray[y * target.width + x] = t / target.channels;
        }
    }

    // ── 2. Downscale to 50% ───────────────────────────────────────────────────
    std::cout << "Downscaling...\n";

    std::vector<float> ref_small, tgt_small;
    int sw, sh;
    downscale_half(ref_gray,  ref.width,    ref.height,    ref_small, sw, sh);
    downscale_half(tgt_gray,  target.width, target.height, tgt_small, sw, sh);

    // Wrap small buffers in Image structs for compute_error at half-res
    Image ref_s(sw, sh, 1);  ref_s.data = ref_small;
    Image tgt_s(sw, sh, 1);  tgt_s.data = tgt_small;

    // ── 3. Multi-patch coarse alignment at half-res ───────────────────────────
    std::cout << "Patch alignment...\n";

    int smallPatch  = std::min(64, std::min(sw, sh) / 4);
    int patchRadius = 100;

    std::vector<Patch> patches =
        selectDistributedPatches(ref_small, sw, sh, smallPatch, 9);   // was 5

    std::vector<int> pdxs, pdys;
    pdxs.reserve(patches.size());
    pdys.reserve(patches.size());

    for (const auto& p : patches)
    {
        IntShift s = estimateShiftPatch(
            ref_small, tgt_small, sw, sh, p, patchRadius);
        pdxs.push_back(s.dx);
        pdys.push_back(s.dy);
    }

    // Median of patch shifts — robust against one bad patch
    std::sort(pdxs.begin(), pdxs.end());
    std::sort(pdys.begin(), pdys.end());
    float rough_dx = pdxs[pdxs.size() / 2] * 2.0f;   // half-res → full-res
    float rough_dy = pdys[pdys.size() / 2] * 2.0f;

    std::cout << "Rough shift (multi-patch median): "
              << rough_dx << "," << rough_dy << "\n";

    // ── 4. Coarse rotation + translation search (runs at half-res) ───────────
    std::cout << "Refinement...\n";

    float best_dx    = rough_dx;
    float best_dy    = rough_dy;
    float best_theta = 0.0f;
    float best_error = std::numeric_limits<float>::max();

    const float T_C_RANGE = 0.035f;   // ±~2 degrees
    const float T_C_STEP  = 0.010f;   //  ~0.57 degrees per step
    const float P_C_RANGE = 3.0f;     // ±3 pixels (full-res)
    const float P_C_STEP  = 0.5f;

    for (float theta = -T_C_RANGE; theta <= T_C_RANGE + 1e-6f; theta += T_C_STEP)
    {
        for (float dy = rough_dy - P_C_RANGE;
             dy <= rough_dy + P_C_RANGE + 1e-6f; dy += P_C_STEP)
        {
            for (float dx = rough_dx - P_C_RANGE;
                 dx <= rough_dx + P_C_RANGE + 1e-6f; dx += P_C_STEP)
            {
                // Half-res: shift is halved, theta is resolution-invariant
                float err = compute_error(ref_s, tgt_s,
                                          dx * 0.5f, dy * 0.5f, theta);
                if (err < best_error)
                {
                    best_error = err;
                    best_dx    = dx;
                    best_dy    = dy;
                    best_theta = theta;
                }
            }
        }
    }

    std::cout << "After refinement: dx=" << best_dx
              << " dy=" << best_dy
              << " theta=" << best_theta << "\n";

    // ── 5. Subpixel refinement at full resolution ─────────────────────────────
    std::cout << "Subpixel refinement...\n";

    float final_dx    = best_dx;
    float final_dy    = best_dy;
    float final_theta = best_theta;
    float final_error = compute_error(ref, target,
                                      final_dx, final_dy, final_theta);

    const float T_F_RANGE = 0.008f;   // ±~0.46 degrees
    const float T_F_STEP  = 0.002f;
    const float P_F_RANGE = 1.0f;     // ±1 pixel
    const float P_F_STEP  = 0.25f;

    for (float theta = best_theta - T_F_RANGE;
         theta <= best_theta + T_F_RANGE + 1e-6f; theta += T_F_STEP)
    {
        for (float dy = best_dy - P_F_RANGE;
             dy <= best_dy + P_F_RANGE + 1e-6f; dy += P_F_STEP)
        {
            for (float dx = best_dx - P_F_RANGE;
                 dx <= best_dx + P_F_RANGE + 1e-6f; dx += P_F_STEP)
            {
                float err = compute_error(ref, target, dx, dy, theta);
                if (err < final_error)
                {
                    final_error = err;
                    final_dx    = dx;
                    final_dy    = dy;
                    final_theta = theta;
                }
            }
        }
    }

    Shift s;
    s.dx    = final_dx;
    s.dy    = final_dy;
    s.theta = final_theta;

    std::cout << "Final shift: dx=" << s.dx
              << " dy=" << s.dy
              << " theta=" << s.theta << "\n";

    return s;
}