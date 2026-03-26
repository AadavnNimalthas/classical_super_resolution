cmake_minimum_required(VERSION 3.26)
project(classical_super_resolution)

set(CMAKE_CXX_STANDARD 17)

# ── ALL source files ─────────────────────────────────────────────────────────
set(SOURCES
        main.cpp

        core/lanczos.cpp
        core/lanczos.h
        core/image.h
        core/upscale.cpp
        core/upscale.h

        io/image_loader.cpp
        io/image_loader.h
        io/image_writer.cpp
        io/image_writer.h
        io/stb_impl.cpp

        alignment/align.cpp
        alignment/align.h

        reconstruction/reconstruct.cpp
        reconstruction/reconstruct.h

        reconstruction/metal_reconstruct.mm
        reconstruction/metal_reconstruct.h

        reconstruction/opencl_reconstruct.cpp
        reconstruction/opencl_reconstruct.h

        utils/math_utils.cpp
        utils/math_utils.h
        utils/system_info.cpp
        utils/system_info.h
)

# ══════════════════════════════════════════════════════════════════════════════
#  APPLE → Metal GPU + OpenMP
# ══════════════════════════════════════════════════════════════════════════════
if(APPLE)

    enable_language(OBJCXX)

    add_executable(classical_super_resolution ${SOURCES})

    # ❌ REMOVE OpenCL files from build on Apple
    set_source_files_properties(
            reconstruction/opencl_reconstruct.cpp
            PROPERTIES HEADER_FILE_ONLY TRUE
    )

    # ── OpenMP (Homebrew) ────────────────────────────────────────────────
    execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
    )

    if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
        message(STATUS "[OpenMP] Homebrew libomp: ${LIBOMP_PREFIX}")

        target_compile_options(classical_super_resolution PRIVATE
                -Xpreprocessor -fopenmp
                -I${LIBOMP_PREFIX}/include
        )

        target_link_libraries(classical_super_resolution PRIVATE
                ${LIBOMP_PREFIX}/lib/libomp.dylib
        )
    else()
        message(WARNING "[OpenMP] libomp not found — CPU will be single-threaded")
    endif()

    # ── Metal ────────────────────────────────────────────────────────────
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

    target_link_libraries(classical_super_resolution PRIVATE
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
    )

    target_compile_definitions(classical_super_resolution PRIVATE USE_METAL=1)

    message(STATUS "[GPU] Backend: Metal")

    # ══════════════════════════════════════════════════════════════════════════════
    #  WINDOWS / LINUX → OpenCL + OpenMP
    # ══════════════════════════════════════════════════════════════════════════════
else()

    add_executable(classical_super_resolution ${SOURCES})

    # ── OpenMP ─────────────────────────────────────────────────────────────
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(classical_super_resolution PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "[OpenMP] Enabled")
    else()
        message(WARNING "[OpenMP] Not found")
    endif()

    # ── OpenCL ─────────────────────────────────────────────────────────────
    find_package(OpenCL)

    if(OpenCL_FOUND)
        target_include_directories(classical_super_resolution PRIVATE
                ${OpenCL_INCLUDE_DIRS}
        )

        target_link_libraries(classical_super_resolution PRIVATE
                ${OpenCL_LIBRARIES}
        )

        target_compile_definitions(classical_super_resolution PRIVATE USE_OPENCL=1)

        message(STATUS "[GPU] Backend: OpenCL")
    else()
        message(WARNING "[GPU] OpenCL not found")
    endif()

endif()