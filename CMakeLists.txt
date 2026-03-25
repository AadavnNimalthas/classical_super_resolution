cmake_minimum_required(VERSION 3.26)
project(classical_super_resolution)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES "*.cpp")

add_executable(classical_super_resolution ${SOURCES}
        core/lanczos.cpp
        core/lanczos.h
        core/image.h
        core/upscale.cpp
        core/upscale.h
        io/image_loader.cpp
        io/image_loader.h
        io/image_writer.cpp
        io/image_writer.h
        alignment/align.cpp
        alignment/align.h
        reconstruction/reconstruct.cpp
        reconstruction/reconstruct.h
        utils/math_utils.h
        utils/math_utils.cpp
        utils/system_info.cpp
        utils/system_info.h
)