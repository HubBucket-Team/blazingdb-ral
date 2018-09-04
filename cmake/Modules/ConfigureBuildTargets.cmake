#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

macro(CONFIGURE_DEFAULT_COMPILER_FLAGS)
    # Host compiler flags
    LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11")

    # Device (GPU) compiler flags
    LIST(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_${CUDA_COMPUTE_CAPABILITY},code=compute_${CUDA_COMPUTE_CAPABILITY}") # virtual architecture (code=compute_X)
    LIST(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_${CUDA_COMPUTE_CAPABILITY},code=sm_${CUDA_COMPUTE_CAPABILITY}") # real architecture (code=sm_X)
    LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math -prec-div false -prec-sqrt false -fmad false")
    LIST(APPEND CUDA_NVCC_FLAGS "--cudart static --relocatable-device-code=false")

    message(STATUS "Default C++ compiler flags for all targets: ${CMAKE_CXX_FLAGS}")
    message(STATUS "Default CUDA compiler flags for all targets: ${CUDA_NVCC_FLAGS}")
endmacro()


macro(DEFINE_SIMPLICITYCONFIG_TARGET)
    # This target will update the Simplicity git information in the header git-config-simplicity.h, see config-simplicity.h.cmake
    add_custom_target(SimplicityConfig COMMAND /bin/bash update-config-simplicity.sh WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endmacro()


macro(DEFINE_FLATBUFFERSSCHEMAS_TARGET)
    # This target will compile FlatBuffers schema files to *_generated.h headers and its binary format
    set(target_name "FlatBuffersSchemas")
    set(schema_dirs "${PROJECT_SOURCE_DIR}/flatbuffers")

    # build_flatbuffers arguments
    set(flatbuffers_schemas "${schema_dirs}/*.fbs")
    set(schema_include_dirs "")
    set(custom_target_name "${target_name}")
    set(additional_dependencies "")
    set(generated_includes_dir "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${target_name}.dir/")
    set(binary_schemas_dir "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${target_name}.dir/")
    set(copy_text_schemas_dir "")

    build_flatbuffers("${flatbuffers_schemas}" "${schema_include_dirs}" "${custom_target_name}" "${additional_dependencies}" "${generated_includes_dir}" "${binary_schemas_dir}" "${copy_text_schemas_dir}")

    # This target will clear the generated sources fileas each time the user invoke the target FlatBuffersSchemas
    add_custom_target(CleanFlatBuffersSchemas COMMAND rm -f *_generated.h && rm -f *.bfbs WORKING_DIRECTORY ${generated_includes_dir})
    add_dependencies(FlatBuffersSchemas CleanFlatBuffersSchemas)

    include_directories("${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${target_name}.dir")
endmacro()


#NOTE
#cpp_flags: host (CPU) compiler flags (for g++)
#cuda_flags: device (GPU) compiler flags (for nvcc, ptxas, cudafe)
#preprocessor_definitions: precompiler definitions (will be used by the host compiler and the device compiler through nvcc directly)
function(configure_target target cpp_flags cuda_flags preprocessor_definitions)
    message(STATUS "Additional compiler flags for build target ${target}. C++ flags: ${cpp_flags} | CUDA flags: ${cuda_flags} | Preprocessor definitions: ${preprocessor_definitions}")

    set(ALL_DEFINITIONS "-DSIMPLICITY_RELEASE_VALUE=${target} ${preprocessor_definitions}")

    cuda_add_executable(${target} ${Simplicity_SRCS} OPTIONS "${cpp_flags} ${cuda_flags} ${ALL_DEFINITIONS}")
    set_target_properties(${target} PROPERTIES COMPILE_FLAGS "${cpp_flags} ${ALL_DEFINITIONS}")

    # Each time we build the target we update the config-simplicity.h file and thus always have the last git commit hash (without run cmake again)
    add_dependencies(${target} FlatBuffersSchemas)
endfunction()


#NOTE: convenience functions that wraps configure_target

function(configure_release_target target preprocessor_definitions)
    configure_target(${target} ${HOST_COMPILER_RELEASE_FLAGS} "" "${preprocessor_definitions}")
endfunction()

function(configure_debug_target target preprocessor_definitions)
    configure_target(${target} ${HOST_COMPILER_DEBUG_FLAGS} "" "${preprocessor_definitions}")
endfunction()

function(configure_cuda_debug_target target preprocessor_definitions)
    configure_target(${target} ${HOST_COMPILER_DEBUG_FLAGS} ${DEVICE_COMPILER_DEBUG_FLAGS} "${preprocessor_definitions}")
endfunction()


macro(CONFIGURE_BUILD_TARGETS)
    message(STATUS "******** Configuring build targets ********")

    #IMPORTANT: Unset the CMAKE_BUILD_TYPE flags force CMake use our custom flags for Simplicity targets and avoid nvcc add default flags.
    set(CMAKE_CXX_FLAGS "")
    set(CMAKE_CXX_FLAGS_DEBUG "")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "")
    set(CMAKE_CXX_FLAGS_RELEASE "")

    # Print all include directories (print all -I directories)
    get_directory_property(includes INCLUDE_DIRECTORIES)
    message(STATUS "Include directories: ${includes}")

    define_simplicityconfig_target()
    configure_default_compiler_flags()
    define_flatbuffersschemas_target()

    # NOTE: precompiler definitions
    # -DDEBUG_ENABLED: Simplicity will print internal debug messages
    # -DRUN_WITHOUT_ROOT: Simplicity doesn't need root privileges in order to run.
    # -DCOMMUNITY: Will build the Community Edition, otherwise will build the Enterprise Edition.
    # -DTESTING_HORIZSCALE_ONEGPU: Allows to emulate two GPU devices using only one (HalfGPU flag).

    set(HOST_COMPILER_RELEASE_FLAGS "-O3")
    set(HOST_COMPILER_DEBUG_FLAGS "-g -O0")
    set(DEVICE_COMPILER_DEBUG_FLAGS "-G")

    # Enterprise Edition targets that run without root privileges
    #configure_release_target(Simplicity_ReleaseWithoutRoot "-DRUN_WITHOUT_ROOT")
    configure_debug_target(Simplicity_DebugWithoutRoot "-DDEBUG_ENABLED -DRUN_WITHOUT_ROOT")
    #configure_cuda_debug_target(Simplicity_CudaDebugWithoutRoot "-DDEBUG_ENABLED -DRUN_WITHOUT_ROOT")

    message(STATUS "******** Build targets are ready ********")
endmacro()
