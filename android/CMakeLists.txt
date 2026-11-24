cmake_minimum_required(VERSION 3.10)

project(whisper.cpp)

set(CMAKE_CXX_STANDARD 17)
set(WHISPER_LIB_DIR ${CMAKE_SOURCE_DIR}/whisper)

message(STATUS "[WHISPER_DEBUG] CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "[WHISPER_DEBUG] WHISPER_LIB_DIR: ${WHISPER_LIB_DIR}")

# Get whisper.cpp version
file(READ "${WHISPER_LIB_DIR}/CMakeLists.txt" MAIN_CMAKE_CONTENT)
string(REGEX MATCH "project\\(\"whisper\\.cpp\" VERSION ([0-9]+\\.[0-9]+\\.[0-9]+)\\)" VERSION_MATCH "${MAIN_CMAKE_CONTENT}")
if(CMAKE_MATCH_1)
    set(WHISPER_VERSION ${CMAKE_MATCH_1} PARENT_SCOPE)
else()
    set(WHISPER_VERSION "unknown" PARENT_SCOPE)
endif()

message(STATUS " Whisper version: ${WHISPER_VERSION}")

# Path to external GGML, otherwise uses the copy in whisper.cpp.
option(GGML_HOME "whisper: Path to external GGML source" OFF)

set(
    SOURCE_FILES
    ${WHISPER_LIB_DIR}/src/whisper.cpp
    ${CMAKE_SOURCE_DIR}/jni.c
    ${CMAKE_SOURCE_DIR}/WhisperBridge.cpp
    )

find_library(LOG_LIB log)

# --- ggml (compute kernels) -------------------------------------------------
# ggml is built ONCE as shared libraries (libggml-base.so / libggml-cpu.so /
# libggml.so) and shared by every libwhisper*.so variant below, so the CPU
# flags must be set on ggml itself, not on the whisper targets.
#
# arm64: target ARMv8.2-A with dotprod + fp16 vector arithmetic. This enables
# the vdotq_s32 paths used by quantized (q4/q5/q8) matmuls and the fp16 flash
# attention kernels. Without it, ggml-cpu falls back to the plain ARMv8.0
# baseline. Cores without these extensions (Cortex-A53/A73 only) are rejected
# at runtime by whisper_bridge_cpu_supported() instead of crashing (SIGILL).
if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    # (i8mm, armv8.6-a, was measured: no significant gain on a Pixel 9a.)
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
endif ()

# Keep OpenMP (ggml default): measured ~1.7x faster than ggml's own thread
# pool on a Pixel 9a (Tensor G4), base-q5_1, 4 threads.
set(GGML_OPENMP ON CACHE BOOL "" FORCE)

include(FetchContent)
if (GGML_HOME)
    FetchContent_Declare(ggml SOURCE_DIR ${GGML_HOME})
else()
    FetchContent_Declare(ggml SOURCE_DIR ${WHISPER_LIB_DIR}/ggml)
endif()
FetchContent_MakeAvailable(ggml)

# Always optimize the kernels, whatever the build type. Flutter debug builds
# use CMAKE_BUILD_TYPE=Debug and profile builds use "profile" (a custom type
# with NO optimization flags): without this, ggml runs at -O0 in both modes
# and any performance measurement made there is meaningless.
foreach (ggml_target ggml ggml-base ggml-cpu)
    if (TARGET ${ggml_target})
        target_compile_options(${ggml_target} PRIVATE -O3)
    endif ()
endforeach ()

function(build_library target_name)
    add_library(
        ${target_name}
        SHARED
        ${SOURCE_FILES}
    )

    target_compile_definitions(${target_name} PUBLIC GGML_USE_CPU)
    target_compile_definitions(${target_name} PRIVATE WHISPER_VERSION="${WHISPER_VERSION}")
    if (${ANDROID_ABI} STREQUAL "arm64-v8a")
        # Lets the bridge reject CPUs the ggml kernels were not built for.
        target_compile_definitions(${target_name} PRIVATE WHISPER_REQUIRE_DOTPROD_FP16)
    endif ()

    if (${target_name} STREQUAL "whisper_v8fp16_va")
        target_compile_options(${target_name} PRIVATE -march=armv8.2-a+fp16)
    elseif (${target_name} STREQUAL "whisper_vfpv4")
        target_compile_options(${target_name} PRIVATE -mfpu=neon-vfpv4)
    endif ()

    # whisper.cpp itself (encoder/decoder orchestration) is optimized in every
    # build type too; only the bridge code would benefit from -O0 debugging.
    target_compile_options(${target_name} PRIVATE -O3)

    if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
        target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)

        target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
        target_link_options(${target_name} PRIVATE -Wl,--exclude-libs,ALL)
        target_link_options(${target_name} PRIVATE -flto)
    endif ()

    target_link_libraries(${target_name} ${LOG_LIB} android ggml)
endfunction()

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    build_library("whisper_v8fp16_va")
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    build_library("whisper_vfpv4")
endif ()

build_library("whisper") # Default target

include_directories(${WHISPER_LIB_DIR})
include_directories(${WHISPER_LIB_DIR}/src)
include_directories(${WHISPER_LIB_DIR}/include)
include_directories(${WHISPER_LIB_DIR}/ggml/include)
include_directories(${WHISPER_LIB_DIR}/ggml/src)
include_directories(${WHISPER_LIB_DIR}/ggml/src/ggml-cpu)
