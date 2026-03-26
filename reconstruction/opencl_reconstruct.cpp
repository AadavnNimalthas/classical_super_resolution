// opencl_reconstruct.cpp
//
// GPU back-projection via OpenCL 1.2.
//
// Compatible runtimes (install one for your GPU):
//   NVIDIA  — CUDA Toolkit  (adds OpenCL to the driver stack)
//   AMD     — ROCm (Linux) / Adrenalin / Windows driver
//   Intel   — Intel oneAPI Base Toolkit / NEO runtime
//   Others  — any ICD-registered OpenCL 1.2+ implementation
//
// Two kernels — same gather-style design as the Metal path:
//   gather_corrections : one work-item per HR pixel, reads from all LR frames
//   apply_corrections  : one work-item per float, applies weighted delta
//
// Both are compiled from a string at runtime so there is no offline toolchain
// dependency (no clang, no Metal compiler, no nvcc).

#include "opencl_reconstruct.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <string>

// ── Platform-portable OpenCL header ──────────────────────────────────────────
#ifdef __APPLE__
  #include <OpenCL/opencl.h>        // Apple ships it as a framework
#else
  #include <CL/cl.h>                // standard Khronos header everywhere else
#endif

// ─────────────────────────────────────────────────────────────────────────────
// OpenCL C kernel source
// ─────────────────────────────────────────────────────────────────────────────
static const char* kKernelSrc = R"CLC(

typedef struct {
    int low_width;
    int low_height;
    int high_width;
    int high_height;
    int channels;
    int scale;
    int num_images;
} ImageParams;

// ── Kernel 1: gather ─────────────────────────────────────────────────────────
// Global size: high_width * high_height * channels
// Each work-item owns one (hr_pixel, channel) pair and gathers contributions
// from every LR frame — no atomics, no race conditions.
__kernel void gather_corrections(
    __global const float*       high_data,
    __global const float*       all_lows,      // all LR frames, concatenated
    __global const float*       shifts,         // [num_images * 2] : dx0,dy0, dx1,dy1 ...
    __global       float*       accum,
    __global       int*         cnt,
    __constant     ImageParams* p
) {
    int gid    = get_global_id(0);
    int total  = p->high_width * p->high_height * p->channels;
    if (gid >= total) return;

    int c   =  gid % p->channels;
    int pix = gid / p->channels;
    int hx  =  pix % p->high_width;
    int hy  =  pix / p->high_width;

    int low_frame_stride = p->low_width * p->low_height * p->channels;

    float total_w_err = 0.0f;
    int   total_cnt   = 0;

    for (int k = 0; k < p->num_images; k++) {
        float dx = shifts[k * 2 + 0];
        float dy = shifts[k * 2 + 1];

        // Inverse-project HR pixel back to LR space
        float lx_f = (float)hx / (float)p->scale - dx;
        float ly_f = (float)hy / (float)p->scale - dy;

        int lx0 = (int)floor(lx_f);
        int ly0 = (int)floor(ly_f);

        for (int dly = 0; dly <= 1; dly++) {
            for (int dlx = 0; dlx <= 1; dlx++) {
                int lx = lx0 + dlx;
                int ly = ly0 + dly;
                if (lx < 0 || lx >= p->low_width)  continue;
                if (ly < 0 || ly >= p->low_height) continue;

                // Forward-project to confirm this LR pixel reaches (hx, hy)
                float fx = ((float)lx + dx) * (float)p->scale;
                float fy = ((float)ly + dy) * (float)p->scale;
                int x0 = (int)fx;
                int y0 = (int)fy;
                if (hx < x0 || hx > x0 + 1) continue;
                if (hy < y0 || hy > y0 + 1) continue;

                float wx = 1.0f - fabs(fx - (float)hx);
                float wy = 1.0f - fabs(fy - (float)hy);
                float w  = wx * wy;
                if (w <= 0.0f) continue;

                float predicted = high_data[gid];
                int   lidx      = k * low_frame_stride
                                + (ly * p->low_width + lx) * p->channels + c;
                float actual    = all_lows[lidx];

                total_w_err += w * (actual - predicted);
                total_cnt++;
            }
        }
    }

    accum[gid] = total_w_err;
    cnt  [gid] = total_cnt;
}

// ── Kernel 2: apply ───────────────────────────────────────────────────────────
// Global size: high_width * high_height * channels
__kernel void apply_corrections(
    __global       float* high_data,
    __global const float* accum,
    __global const int*   cnt,
             float        alpha
) {
    int gid = get_global_id(0);
    if (cnt[gid] > 0) {
        high_data[gid] += alpha * (accum[gid] / (float)cnt[gid]);
    }
}

)CLC";

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static const char* cl_err_str(cl_int e) {
    switch(e) {
        case CL_SUCCESS:                         return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_BUILD_PROGRAM_FAILURE:           return "CL_BUILD_PROGRAM_FAILURE";
        case CL_OUT_OF_RESOURCES:                return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:              return "CL_OUT_OF_HOST_MEMORY";
        case CL_INVALID_PROGRAM_EXECUTABLE:      return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_ARGS:             return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_GROUP_SIZE:         return "CL_INVALID_WORK_GROUP_SIZE";
        default: return "CL_UNKNOWN_ERROR";
    }
}

#define CL_CHECK(expr)                                                      \
    do {                                                                    \
        cl_int _e = (expr);                                                 \
        if (_e != CL_SUCCESS) {                                             \
            std::cerr << "[OpenCL] " #expr " failed: "                      \
                      << cl_err_str(_e) << " (" << _e << ")\n";            \
            return false;                                                   \
        }                                                                   \
    } while(0)

// ── Device picker ─────────────────────────────────────────────────────────────
// Priority: discrete GPU > integrated GPU > CPU device
// Tries all platforms so it works regardless of vendor.
static bool pick_device(cl_platform_id& out_platform, cl_device_id& out_device) {
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    if (num_platforms == 0) {
        std::cerr << "[OpenCL] No platforms found.\n";
        return false;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    // Collect every GPU device across all platforms
    struct Candidate { cl_platform_id plat; cl_device_id dev; cl_device_type type; };
    std::vector<Candidate> candidates;

    for (auto& plat : platforms) {
        // Try GPU first, then CPU as fallback device type
        for (cl_device_type dtype : { CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU }) {
            cl_uint n = 0;
            if (clGetDeviceIDs(plat, dtype, 0, nullptr, &n) != CL_SUCCESS || n == 0)
                continue;
            std::vector<cl_device_id> devs(n);
            clGetDeviceIDs(plat, dtype, n, devs.data(), nullptr);
            for (auto& d : devs)
                candidates.push_back({plat, d, dtype});
        }
    }

    if (candidates.empty()) {
        std::cerr << "[OpenCL] No devices found across " << num_platforms << " platform(s).\n";
        return false;
    }

    // Prefer discrete GPU (CL_DEVICE_HOST_UNIFIED_MEMORY == false) over
    // integrated/CPU. Falls back gracefully if the extension is absent.
    cl_bool best_unified = CL_TRUE;
    out_platform = candidates[0].plat;
    out_device   = candidates[0].dev;

    for (auto& c : candidates) {
        if (c.type != CL_DEVICE_TYPE_GPU) continue;  // prefer GPU over CPU device
        cl_bool unified = CL_TRUE;
        clGetDeviceInfo(c.dev, CL_DEVICE_HOST_UNIFIED_MEMORY,
                        sizeof(unified), &unified, nullptr);
        if (!unified || best_unified) {   // discrete GPU wins
            out_platform  = c.plat;
            out_device    = c.dev;
            best_unified  = unified;
            if (!unified) break;          // discrete GPU found — stop searching
        }
    }

    // Print chosen device name
    char name[256] = {};
    clGetDeviceInfo(out_device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    char vendor[256] = {};
    clGetDeviceInfo(out_device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
    std::cout << "[OpenCL] Using: " << vendor << " — " << name << "\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────
bool opencl_reconstruct(
    const std::vector<Image>& images,
    const std::vector<Shift>& shifts,
    Image& high,
    int    scale,
    float  alpha,
    int    iterations
) {
    // ── Select platform + device ──────────────────────────────────────────────
    cl_platform_id platform;
    cl_device_id   device;
    if (!pick_device(platform, device)) return false;

    // ── Context + command queue ───────────────────────────────────────────────
    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCL] clCreateContext failed: " << cl_err_str(err) << "\n";
        return false;
    }

    // CL_QUEUE_PROFILING_ENABLE is optional but harmless
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCL] clCreateCommandQueue failed: " << cl_err_str(err) << "\n";
        clReleaseContext(ctx);
        return false;
    }

    // ── Build program ─────────────────────────────────────────────────────────
    cl_program program = clCreateProgramWithSource(ctx, 1, &kKernelSrc, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCL] clCreateProgramWithSource failed: " << cl_err_str(err) << "\n";
        clReleaseCommandQueue(queue); clReleaseContext(ctx);
        return false;
    }

    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2 -cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Print the build log so the user can debug
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "[OpenCL] Build error:\n" << log << "\n";
        clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(ctx);
        return false;
    }

    cl_kernel k_gather = clCreateKernel(program, "gather_corrections", &err);
    if (err != CL_SUCCESS) { std::cerr << "[OpenCL] kernel 'gather_corrections' not found\n"; return false; }
    cl_kernel k_apply  = clCreateKernel(program, "apply_corrections",  &err);
    if (err != CL_SUCCESS) { std::cerr << "[OpenCL] kernel 'apply_corrections' not found\n";  return false; }

    // ── Pack all LR frames ────────────────────────────────────────────────────
    int low_w  = images[0].width;
    int low_h  = images[0].height;
    int chans  = images[0].channels;
    int frame_floats  = low_w * low_h * chans;
    int total_low_len = (int)images.size() * frame_floats;

    std::vector<float> all_lows(total_low_len);
    for (int k = 0; k < (int)images.size(); k++)
        std::memcpy(all_lows.data() + k * frame_floats,
                    images[k].data.data(),
                    frame_floats * sizeof(float));

    // Pack shifts as flat float array: dx0, dy0, dx1, dy1, ...
    std::vector<float> shift_flat(images.size() * 2);
    for (int k = 0; k < (int)images.size(); k++) {
        shift_flat[k * 2 + 0] = shifts[k].dx;
        shift_flat[k * 2 + 1] = shifts[k].dy;
    }

    int high_len = (int)high.data.size();   // high_w * high_h * channels

    struct ImageParams {
        cl_int low_width, low_height;
        cl_int high_width, high_height;
        cl_int channels, scale, num_images;
    } params {
        low_w, low_h,
        high.width, high.height,
        chans, scale, (int)images.size()
    };

    // ── Allocate GPU buffers ──────────────────────────────────────────────────
    cl_mem buf_high   = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                       high_len * sizeof(float), nullptr, &err);
    cl_mem buf_lows   = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       total_low_len * sizeof(float), all_lows.data(), &err);
    cl_mem buf_shifts = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       shift_flat.size() * sizeof(float), shift_flat.data(), &err);
    cl_mem buf_accum  = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                       high_len * sizeof(float), nullptr, &err);
    cl_mem buf_cnt    = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                       high_len * sizeof(cl_int), nullptr, &err);
    cl_mem buf_params = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(params), &params, &err);

    if (!buf_high || !buf_lows || !buf_shifts || !buf_accum || !buf_cnt || !buf_params) {
        std::cerr << "[OpenCL] Buffer allocation failed.\n";
        return false;
    }

    // ── Set gather kernel args (args 0-5 are constant across iterations) ──────
    CL_CHECK(clSetKernelArg(k_gather, 0, sizeof(cl_mem), &buf_high));
    CL_CHECK(clSetKernelArg(k_gather, 1, sizeof(cl_mem), &buf_lows));
    CL_CHECK(clSetKernelArg(k_gather, 2, sizeof(cl_mem), &buf_shifts));
    CL_CHECK(clSetKernelArg(k_gather, 3, sizeof(cl_mem), &buf_accum));
    CL_CHECK(clSetKernelArg(k_gather, 4, sizeof(cl_mem), &buf_cnt));
    CL_CHECK(clSetKernelArg(k_gather, 5, sizeof(cl_mem), &buf_params));

    // ── Set apply kernel args ─────────────────────────────────────────────────
    CL_CHECK(clSetKernelArg(k_apply, 0, sizeof(cl_mem), &buf_high));
    CL_CHECK(clSetKernelArg(k_apply, 1, sizeof(cl_mem), &buf_accum));
    CL_CHECK(clSetKernelArg(k_apply, 2, sizeof(cl_mem), &buf_cnt));
    CL_CHECK(clSetKernelArg(k_apply, 3, sizeof(float),  &alpha));

    // ── Determine work-group size (device-safe) ───────────────────────────────
    size_t max_wg = 256;
    clGetKernelWorkGroupInfo(k_gather, device, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(max_wg), &max_wg, nullptr);
    // Round down to a power of two for safety
    size_t wg = 1;
    while (wg * 2 <= max_wg) wg *= 2;

    size_t global_size = (size_t)high_len;
    // Round up to multiple of wg
    if (global_size % wg != 0)
        global_size = (global_size / wg + 1) * wg;

    // ── Iterative refinement ──────────────────────────────────────────────────
    for (int iter = 0; iter < iterations; iter++) {
        std::cout << "Iteration " << iter << " (OpenCL GPU)\n";

        // Upload current HR estimate
        CL_CHECK(clEnqueueWriteBuffer(queue, buf_high, CL_TRUE, 0,
                                      high_len * sizeof(float),
                                      high.data.data(), 0, nullptr, nullptr));

        // Zero accum + cnt
        float  zero_f = 0.0f;
        cl_int zero_i = 0;
        CL_CHECK(clEnqueueFillBuffer(queue, buf_accum, &zero_f, sizeof(float),
                                     0, high_len * sizeof(float), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueFillBuffer(queue, buf_cnt,   &zero_i, sizeof(cl_int),
                                     0, high_len * sizeof(cl_int), 0, nullptr, nullptr));

        // Pass 1: gather
        CL_CHECK(clEnqueueNDRangeKernel(queue, k_gather, 1,
                                        nullptr, &global_size, &wg,
                                        0, nullptr, nullptr));

        // Pass 2: apply
        CL_CHECK(clEnqueueNDRangeKernel(queue, k_apply, 1,
                                        nullptr, &global_size, &wg,
                                        0, nullptr, nullptr));

        // Read back refined HR image
        CL_CHECK(clEnqueueReadBuffer(queue, buf_high, CL_TRUE, 0,
                                     high_len * sizeof(float),
                                     high.data.data(), 0, nullptr, nullptr));
    }

    CL_CHECK(clFinish(queue));

    // ── Cleanup ───────────────────────────────────────────────────────────────
    clReleaseMemObject(buf_high);
    clReleaseMemObject(buf_lows);
    clReleaseMemObject(buf_shifts);
    clReleaseMemObject(buf_accum);
    clReleaseMemObject(buf_cnt);
    clReleaseMemObject(buf_params);
    clReleaseKernel(k_gather);
    clReleaseKernel(k_apply);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return true;
}
