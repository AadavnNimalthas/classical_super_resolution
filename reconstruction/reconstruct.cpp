#include "reconstruct.h"
#include "../core/upscale.h"
#include "../alignment/align.h"

// Each GPU backend is compiled only on its target platform.
// Exactly one of these will be defined by CMake.
#ifdef USE_METAL
  #include "metal_reconstruct.h"
#endif
#ifdef USE_OPENCL
  #include "opencl_reconstruct.h"
#endif

#include <iostream>
#include <cmath>
#include <algorithm>

Image reconstruct(const std::vector<Image>& images, int scale)
{
    if (images.empty()) {
        std::cerr << "No input images\n";
        std::exit(1);
    }

    const Image& ref = images[0];

    // ── Alignment (serial CPU — fast enough, no parallelism needed) ──────────
    std::cout << "Aligning " << images.size() << " images...\n";
    std::vector<Shift> shifts(images.size());
    for (size_t i = 0; i < images.size(); i++)
        shifts[i] = estimate_shift(ref, images[i]);

    // ── Initial Lanczos upscale (OpenMP-parallel internally) ─────────────────
    std::cout << "Lanczos upscale (OpenMP CPU)...\n";
    Image high = upscale_lanczos(ref, scale, 3);

    const int   iterations = 6;
    const float alpha      = 0.15f;

    // ══════════════════════════════════════════════════════════════════════════
    //  GPU dispatch — try whichever backend was compiled in.
    //  Falls through to CPU if the runtime reports no device.
    // ══════════════════════════════════════════════════════════════════════════

#ifdef USE_METAL
    // macOS: Metal (Apple Silicon + AMD/Intel discrete on Intel Macs)
    if (metal_reconstruct(images, shifts, high, scale, alpha, iterations))
        return high;
    std::cout << "[Metal] No device — falling back to CPU\n";
#endif

#ifdef USE_OPENCL
    // Windows / Linux: OpenCL (NVIDIA, AMD, Intel, any ICD-registered GPU)
    if (opencl_reconstruct(images, shifts, high, scale, alpha, iterations))
        return high;
    std::cout << "[OpenCL] No device — falling back to CPU\n";
#endif

    // ══════════════════════════════════════════════════════════════════════════
    //  CPU fallback — OpenMP parallel back-projection.
    //  Works everywhere, no GPU required.
    // ══════════════════════════════════════════════════════════════════════════
    for (int iter = 0; iter < iterations; iter++) {
        std::cout << "Iteration " << iter << " (CPU/OpenMP)\n";

        std::vector<float> accum(high.data.size(), 0.0f);
        std::vector<int>   cnt  (high.data.size(), 0);

        #pragma omp parallel
        {
            // Thread-private buffers → single critical reduction at the end.
            // Avoids false sharing and per-pixel locking.
            std::vector<float> local_accum(high.data.size(), 0.0f);
            std::vector<int>   local_cnt  (high.data.size(), 0);

            #pragma omp for schedule(dynamic)
            for (int k = 0; k < (int)images.size(); k++) {
                const Image& low = images[k];
                Shift shift = shifts[k];

                for (int y = 0; y < low.height; y++) {
                    for (int x = 0; x < low.width; x++) {
                        float fx = (x + shift.dx) * scale;
                        float fy = (y + shift.dy) * scale;
                        int x0 = (int)fx;
                        int y0 = (int)fy;

                        for (int dy = 0; dy <= 1; dy++) {
                            for (int dx = 0; dx <= 1; dx++) {
                                int hx = x0 + dx;
                                int hy = y0 + dy;
                                if (hx < 0 || hx >= high.width)  continue;
                                if (hy < 0 || hy >= high.height) continue;

                                float wx = 1.0f - std::abs(fx - hx);
                                float wy = 1.0f - std::abs(fy - hy);
                                float w  = wx * wy;
                                if (w <= 0.0f) continue;

                                for (int c = 0; c < low.channels; c++) {
                                    int idx = (hy * high.width + hx) * high.channels + c;
                                    float err = low.at(x, y, c) - high.data[idx];
                                    local_accum[idx] += w * err;
                                    local_cnt  [idx]++;
                                }
                            }
                        }
                    }
                }
            }

            #pragma omp critical
            {
                for (size_t i = 0; i < accum.size(); i++) {
                    accum[i] += local_accum[i];
                    cnt  [i] += local_cnt  [i];
                }
            }
        } // end omp parallel

        for (size_t i = 0; i < high.data.size(); i++)
            if (cnt[i] > 0)
                high.data[i] += alpha * (accum[i] / cnt[i]);
    }

    return high;
}
