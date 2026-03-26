// 1.0
// metal_reconstruct.mm  — Objective-C++ (compiled only on Apple)
//
// Two Metal compute kernels:
//   1. gather_corrections  – one GPU thread per HR pixel; pulls contributions
//      from every LR image with no atomic conflicts (gather, not scatter).
//   2. apply_corrections   – one GPU thread per float in the HR buffer;
//      applies the weighted error correction.

#import "metal_reconstruct.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <iostream>
#import <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// Metal Shading Language source (compiled at runtime — no .metallib needed)
// ─────────────────────────────────────────────────────────────────────────────
static const char* kShaderSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct ImageParams {
    int low_width;
    int low_height;
    int high_width;
    int high_height;
    int channels;
    int scale;
    int num_images;
};

// ── Kernel 1: gather ─────────────────────────────────────────────────────────
// Thread grid: (high_width x high_height).
// For every HR pixel we find which LR pixels (across all frames) map to it,
// compute the weighted error, and accumulate — no race conditions.
kernel void gather_corrections(
    device const float*       high_data  [[ buffer(0) ]],
    device const float*       all_lows   [[ buffer(1) ]],  // all LR frames concatenated
    device const float2*      shifts     [[ buffer(2) ]],  // per-frame (dx, dy)
    device       float*       accum      [[ buffer(3) ]],
    device       int*         cnt        [[ buffer(4) ]],
    constant     ImageParams& p          [[ buffer(5) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    int hx = (int)gid.x;
    int hy = (int)gid.y;
    if (hx >= p.high_width || hy >= p.high_height) return;

    int low_frame_stride = p.low_width * p.low_height * p.channels;

    for (int c = 0; c < p.channels; c++) {
        int hidx = (hy * p.high_width + hx) * p.channels + c;
        float total_w_err = 0.0f;
        int   total_cnt   = 0;

        for (int k = 0; k < p.num_images; k++) {
            float dx = shifts[k].x;
            float dy = shifts[k].y;

            // Inverse-project: which LR coordinate maps to this HR pixel?
            float lx_f = (float)hx / (float)p.scale - dx;
            float ly_f = (float)hy / (float)p.scale - dy;

            int lx0 = (int)floor(lx_f);
            int ly0 = (int)floor(ly_f);

            // Bilinear neighborhood: the 2x2 LR pixels that could contribute
            for (int dly = 0; dly <= 1; dly++) {
                for (int dlx = 0; dlx <= 1; dlx++) {
                    int lx = lx0 + dlx;
                    int ly = ly0 + dly;
                    if (lx < 0 || lx >= p.low_width) continue;
                    if (ly < 0 || ly >= p.low_height) continue;

                    // Forward-project to confirm this LR pixel actually hits (hx, hy)
                    float fx = ((float)lx + dx) * (float)p.scale;
                    float fy = ((float)ly + dy) * (float)p.scale;
                    int x0 = (int)fx;
                    int y0 = (int)fy;
                    if (hx < x0 || hx > x0 + 1) continue;
                    if (hy < y0 || hy > y0 + 1) continue;

                    float wx = 1.0f - abs(fx - (float)hx);
                    float wy = 1.0f - abs(fy - (float)hy);
                    float w  = wx * wy;
                    if (w <= 0.0f) continue;

                    float predicted = high_data[hidx];
                    int   lidx      = k * low_frame_stride
                                    + (ly * p.low_width + lx) * p.channels + c;
                    float actual    = all_lows[lidx];

                    total_w_err += w * (actual - predicted);
                    total_cnt++;
                }
            }
        }

        accum[hidx] = total_w_err;
        cnt  [hidx] = total_cnt;
    }
}

// ── Kernel 2: apply ───────────────────────────────────────────────────────────
// Thread grid: flat index over all floats in the HR image.
kernel void apply_corrections(
    device       float* high_data [[ buffer(0) ]],
    device const float* accum     [[ buffer(1) ]],
    device const int*   cnt       [[ buffer(2) ]],
    constant     float& alpha     [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (cnt[gid] > 0) {
        high_data[gid] += alpha * (accum[gid] / (float)cnt[gid]);
    }
}
)MSL";

// ─────────────────────────────────────────────────────────────────────────────

bool metal_reconstruct(
    const std::vector<Image>& images,
    const std::vector<Shift>& shifts,
    Image& high,
    int    scale,
    float  alpha,
    int    iterations
) {
    @autoreleasepool {

        // ── Device & library ─────────────────────────────────────────────────
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "[Metal] No Metal device found.\n";
            return false;
        }
        std::cout << "[Metal] Using GPU: "
                  << [device.name UTF8String] << "\n";

        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:@(kShaderSrc)
                                                  options:nil
                                                    error:&err];
        if (!lib) {
            std::cerr << "[Metal] Shader compile error: "
                      << [[err localizedDescription] UTF8String] << "\n";
            return false;
        }

        id<MTLComputePipelineState> gatherPSO =
            [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather_corrections"]
                                                  error:&err];
        id<MTLComputePipelineState> applyPSO =
            [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"apply_corrections"]
                                                  error:&err];

        if (!gatherPSO || !applyPSO) {
            std::cerr << "[Metal] PSO creation error: "
                      << [[err localizedDescription] UTF8String] << "\n";
            return false;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        // ── Pack all LR frames into one flat buffer ───────────────────────────
        int low_w  = images[0].width;
        int low_h  = images[0].height;
        int chans  = images[0].channels;
        int frame_floats = low_w * low_h * chans;
        int total_low    = (int)images.size() * frame_floats;

        std::vector<float> all_lows(total_low);
        for (int k = 0; k < (int)images.size(); k++)
            std::memcpy(all_lows.data() + k * frame_floats,
                        images[k].data.data(),
                        frame_floats * sizeof(float));

        // Pack shifts as float2
        std::vector<float> shift_data(images.size() * 2);
        for (int k = 0; k < (int)images.size(); k++) {
            shift_data[k * 2 + 0] = shifts[k].dx;
            shift_data[k * 2 + 1] = shifts[k].dy;
        }

        int high_floats = (int)high.data.size();

        // ── GPU buffers (Shared storage = zero-copy on Apple Silicon) ─────────
        id<MTLBuffer> lowBuf = [device
            newBufferWithBytes:all_lows.data()
                        length:total_low * sizeof(float)
                       options:MTLResourceStorageModeShared];

        id<MTLBuffer> shiftBuf = [device
            newBufferWithBytes:shift_data.data()
                        length:shift_data.size() * sizeof(float)
                       options:MTLResourceStorageModeShared];

        id<MTLBuffer> highBuf = [device
            newBufferWithLength:high_floats * sizeof(float)
                        options:MTLResourceStorageModeShared];

        id<MTLBuffer> accumBuf = [device
            newBufferWithLength:high_floats * sizeof(float)
                        options:MTLResourceStorageModeShared];

        id<MTLBuffer> cntBuf = [device
            newBufferWithLength:high_floats * sizeof(int)
                        options:MTLResourceStorageModeShared];

        struct ImageParams {
            int low_width, low_height;
            int high_width, high_height;
            int channels, scale, num_images;
        } params {
            low_w, low_h,
            high.width, high.height,
            chans, scale, (int)images.size()
        };

        id<MTLBuffer> paramsBuf = [device
            newBufferWithBytes:&params
                        length:sizeof(params)
                       options:MTLResourceStorageModeShared];

        // ── Thread dimensions ─────────────────────────────────────────────────
        // Gather: 2-D grid over HR image
        NSUInteger tw = gatherPSO.threadExecutionWidth;
        NSUInteger th = gatherPSO.maxTotalThreadsPerThreadgroup / tw;
        MTLSize gatherGroup = MTLSizeMake(tw, th, 1);
        MTLSize gatherGrid  = MTLSizeMake(high.width, high.height, 1);

        // Apply: flat 1-D grid
        NSUInteger applyW = applyPSO.maxTotalThreadsPerThreadgroup;
        MTLSize applyGroup = MTLSizeMake(applyW, 1, 1);
        MTLSize applyGrid  = MTLSizeMake(high_floats, 1, 1);

        // ── Iterative refinement ──────────────────────────────────────────────
        for (int iter = 0; iter < iterations; iter++) {
            std::cout << "Iteration " << iter << " (Metal GPU)\n";

            // Upload current HR estimate
            std::memcpy([highBuf contents],
                        high.data.data(),
                        high_floats * sizeof(float));
            std::memset([accumBuf contents], 0, high_floats * sizeof(float));
            std::memset([cntBuf   contents], 0, high_floats * sizeof(int));

            id<MTLCommandBuffer> cmd = [queue commandBuffer];

            // Pass 1: gather
            {
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:gatherPSO];
                [enc setBuffer:highBuf   offset:0 atIndex:0];
                [enc setBuffer:lowBuf    offset:0 atIndex:1];
                [enc setBuffer:shiftBuf  offset:0 atIndex:2];
                [enc setBuffer:accumBuf  offset:0 atIndex:3];
                [enc setBuffer:cntBuf    offset:0 atIndex:4];
                [enc setBuffer:paramsBuf offset:0 atIndex:5];
                [enc dispatchThreads:gatherGrid
                 threadsPerThreadgroup:gatherGroup];
                [enc endEncoding];
            }

            // Pass 2: apply
            {
                float a = alpha;
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:applyPSO];
                [enc setBuffer:highBuf  offset:0 atIndex:0];
                [enc setBuffer:accumBuf offset:0 atIndex:1];
                [enc setBuffer:cntBuf   offset:0 atIndex:2];
                [enc setBytes:&a length:sizeof(float) atIndex:3];
                [enc dispatchThreads:applyGrid
                 threadsPerThreadgroup:applyGroup];
                [enc endEncoding];
            }

            [cmd commit];
            [cmd waitUntilCompleted];

            // Download refined HR image
            std::memcpy(high.data.data(),
                        [highBuf contents],
                        high_floats * sizeof(float));
        }

        return true;
    } // @autoreleasepool
}
