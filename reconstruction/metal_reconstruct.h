#pragma once
#include "../core/image.h"
#include "../alignment/align.h"
#include "reconstruct.h"
#include <vector>

// Runs the full iterative back-projection loop on the Metal GPU.
// Returns true on success; false if Metal is unavailable (caller should CPU-fallback).
bool metal_reconstruct(
    const std::vector<Image>& images,
    const std::vector<Shift>& shifts,
    Image& high,          // in/out: initial Lanczos guess → refined result
    int    scale,
    float  alpha,
    int    iterations
);
