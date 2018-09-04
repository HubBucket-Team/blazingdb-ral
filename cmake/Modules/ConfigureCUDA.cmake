#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#     Copyright 2018 Christian Noboa <christian@blazingdb.com>
#=============================================================================

# BEGIN macros

# check the cmake arg -DGPU_COMPUTE_CAPABILITY and if not present defines its default value
macro(CONFIGURE_GPU_COMPUTE_CAPABILITY)
    #GPU_COMPUTE_CAPABILITY 30 means GPU compute capability version 3.0

    if(NOT GPU_COMPUTE_CAPABILITY)
        message(AUTHOR_WARNING "NVIDIA GPU Compute Capability is not defined, using NVIDIA GPU compute capability version 3.0")
        set(GPU_COMPUTE_CAPABILITY "30")
    endif ()
endmacro()

macro(CONFIGURE_CUDA_LIBRARIES)
    set(CUDA_LIBRARY_DIR       ${CUDA_TOOLKIT_ROOT_DIR})
    set(CUDA_LIBRARY_STUBS_DIR ${CUDA_TOOLKIT_ROOT_DIR})

    ## Based on NVRTC (Runtime Compilation) - CUDA Toolkit Documentation - v9.2.148
    ## 2.2. Installation
    if(CMAKE_HOST_APPLE)
        set(CUDA_LIBRARY_DIR       ${CUDA_TOOLKIT_ROOT_DIR}/lib)
        set(CUDA_LIBRARY_STUBS_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs)
    elseif(CMAKE_HOST_UNIX)
        set(CUDA_LIBRARY_DIR       ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
        set(CUDA_LIBRARY_STUBS_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
    elseif(CMAKE_HOST_WIN32)
        set(CUDA_LIBRARY_DIR       ${CUDA_TOOLKIT_ROOT_DIR}\lib\x64)
        set(CUDA_LIBRARY_STUBS_DIR ${CUDA_TOOLKIT_ROOT_DIR}\lib\x64\stubs)
    endif()

    set(CUDA_CUDA_LIBRARY  cuda)
    set(CUDA_NVRTC_LIBRARY nvrtc)

    message(STATUS "CUDA_CUDA_LIBRARY: ${CUDA_CUDA_LIBRARY}")
    message(STATUS "CUDA_NVRTC_LIBRARY: ${CUDA_NVRTC_LIBRARY}")
    message(STATUS "CUDA_LIBRARY_DIR: ${CUDA_LIBRARY_DIR}")
    message(STATUS "CUDA_LIBRARY_STUBS_DIR: ${CUDA_LIBRARY_STUBS_DIR}")

    # TODO percy seems cmake bug: we cannot define target dirs per cuda target
    # ... see if works in future cmake versions
    link_directories(${CUDA_LIBRARY_DIR})
    link_directories(${CUDA_LIBRARY_STUBS_DIR})
endmacro()

# compute_capability is a int value (e.g. 30 means compute capability 3.0)
macro(CONFIGURE_CUDA_COMPILER compute_capability)
    include_directories(${CUDA_INCLUDE_DIRS})

    # Host compiler flags
    # WARNING never add "-std=c++11" to APPEND CMAKE_CXX_FLAGS since it will be redundant and causes build issues
    #list(APPEND CMAKE_CXX_FLAGS "")

    # Device (GPU) compiler flags
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
    list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_${compute_capability},code=compute_${compute_capability}") # virtual architecture (code=compute_X)
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_${compute_capability},code=sm_${compute_capability}") # real architecture (code=sm_X)
    #list(APPEND CUDA_NVCC_FLAGS "--use_fast_math -prec-div false -prec-sqrt false -fmad false")
    #list(APPEND CUDA_NVCC_FLAGS "--cudart static --relocatable-device-code=false")
    #list(APPEND CUDA_NVCC_FLAGS "--default-stream per-thread")

    message(STATUS "Default C++ compiler flags for all targets: ${CMAKE_CXX_FLAGS}")
    message(STATUS "Default CUDA compiler flags for all targets: ${CUDA_NVCC_FLAGS}")
endmacro()

# END macros


# BEGIN MAIN #

set(CUDA_SDK_ROOT_DIR "/usr/local/cuda") # /usr/local/cuda is the standard installation directory
find_package(CUDA REQUIRED)
set_package_properties(CUDA PROPERTIES TYPE REQUIRED
    PURPOSE "NVIDIA CUDA® parallel computing platform and programming model."
    URL "https://developer.nvidia.com/cuda-zone"
)

if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found, please check your settings.")
endif()

message(STATUS "CUDA ${CUDA_VERSION} found in ${CUDA_TOOLKIT_ROOT_DIR}")

configure_gpu_compute_capability()
configure_cuda_libraries()
configure_cuda_compiler(${GPU_COMPUTE_CAPABILITY})

# END MAIN #
