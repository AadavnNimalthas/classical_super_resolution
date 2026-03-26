// 1.0
#pragma once
#include "../core/image.h"
#include "../alignment/align.h"
#include "reconstruct.h"
#include <vector>

// Runs the iterative back-projection loop on the best available OpenCL device.
// Works on any GPU (NVIDIA / AMD / Intel / other) that has an OpenCL 1.2+ runtime.
//
// Returns true on success.
// Returns false if no OpenCL platform is found or initialisation fails
// — caller should fall back to the CPU path.
bool opencl_reconstruct(
    const std::vector<Image>& images,
    const std::vector<Shift>& shifts,
    Image& high,          // in/out: Lanczos initial guess → refined result
    int    scale,
    float  alpha,
    int    iterations
);
