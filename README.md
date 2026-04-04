**Classical Super-Resolution (C++)**

v1.0

A cross-platform classical super-resolution pipeline that reconstructs a higher-quality image from multiple inputs using multi-threaded CPU processing and GPU acceleration.

There is a current algorithm bug in alignment, shifts are noted however the shift does not happen to the images.

**Features**

Multi-threaded image alignment (OpenMP)
Multi-scale (pyramid) alignment for large image shifts
Subpixel alignment using bilinear interpolation
GPU-accelerated reconstruction
Metal (macOS)
OpenCL (Windows / Linux)
High-quality Lanczos upscaling
Cross-platform support (macOS, Linux, Windows)
CPU fallback support (works without GPU)

**Release 1.0**

Completed full classical super-resolution pipeline
Implemented multi-scale alignment for large motion
Added subpixel refinement for higher precision
Reduced alignment runtime significantly (10×–30× faster)
Fixed ghosting and major misalignment artifacts
Integrated GPU reconstruction (Metal / OpenCL)
Added cross-platform build support

**Pipeline Overview**

Load input images
Select best reference image
Align images (CPU, multi-threaded)
Reconstruct image (GPU if available)
Upscale using Lanczos
Output final high-resolution image

**Requirements**

**All Platforms:**

CMake 3.26 or higher
C++17 compatible compiler

**macOS:**

Xcode / Apple Clang
Install OpenMP:
brew install libomp

**Linux:**

Install dependencies:
sudo apt install build-essential cmake libomp-dev ocl-icd-opencl-dev opencl-headers

**Windows:**

Visual Studio (MSVC)
CMake
GPU drivers (optional for OpenCL support)

**Build Instructions**

**macOS / Linux:**

(Optional clean rebuild)
rm -rf build

mkdir build
cd build
cmake ..

**macOS:**
make -j$(sysctl -n hw.ncpu)

**Linux:**
make -j$(nproc)

**Windows (Command Prompt):**

(Optional clean rebuild)
rmdir /s /q build

mkdir build
cd build
cmake ..
cmake --build . --config Release

**Run**

**macOS / Linux:**
./classical_super_resolution

**Windows:**
classical_super_resolution.exe

**Input**
When prompted, enter the full path to your image folder:

**macOS:**
/Users/username/path_to_folder

**Linux:**
/home/username/path_to_folder

**Windows:**
C:\Users\username\path_to_folder

**Output**

Outputs a reconstructed and upscaled image (e.g., output.png)
Combines information from multiple input images
Enhances detail and reduces noise

**Configuration**

Search radius (alignment): affects accuracy vs speed
Upscale factor: typically 2x or 4x

**Performance Notes**

Alignment is the most computationally expensive stage
Multi-threading significantly improves performance
Multi-scale alignment reduces runtime dramatically
GPU acceleration improves reconstruction speed
Program works on CPU-only systems with reduced performance

**Known Limitations**

Alignment may fail if images differ too much
Very large images (30MP+) increase runtime
GPU acceleration depends on driver support
No automatic exposure or brightness correction

**Future Improvements**

Multi-level pyramid alignment (more levels)
FFT / phase correlation alignment
Exposure and color normalization
GPU-based upscaling
Video super-resolution support

**Technologies Used**

C++17
OpenMP (CPU parallelism)
Metal (macOS GPU acceleration)
OpenCL (cross-platform GPU acceleration)
CMake (build system)

**Key Concepts**

Image alignment and registration
Reconstruction-based super-resolution
Subpixel accuracy
Lanczos resampling
Parallel computing (CPU + GPU)

**Author**

**Aadavn Nimalthas**
