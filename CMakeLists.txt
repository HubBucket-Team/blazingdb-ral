#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

cmake_minimum_required(VERSION 3.11)

project(testing-cmake-libgdf)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Using C++ standard: c++${CMAKE_CXX_STANDARD}")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_MODULE_PATH:" "${CMAKE_MODULE_PATH}")

# Include CMake modules
include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)

# NOTE Do not change the inclusion order
# Configure dependencies
include(ConfigureCUDA)
include(ConfigureLibGDF)
include(ConfigureFlatBuffers)
include(ConfigureBlazingDBProtocol)
#include(ConfigureArrow)
include(ConfigureGoogleTest)

# BEGIN MAIN #
include_directories(
    ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/thirdparty/jitify
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper)

## Binary Operations
add_subdirectory(${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation)
get_directory_property(binary_operation_source_files DIRECTORY ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation DEFINITION source_files)

set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/CalciteInterpreter.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/ColumnManipulation.cu
              ${CMAKE_CURRENT_SOURCE_DIR}/src/ResultSetRepository.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/JoinProcessor.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/LogicalFilter.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/StringUtil.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/QueryState.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/CodeTimer.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/Utils.cu
              ${CMAKE_CURRENT_SOURCE_DIR}/src/GDFCounter.cu
              ${CMAKE_CURRENT_SOURCE_DIR}/src/GDFColumn.cu
              ${CMAKE_CURRENT_SOURCE_DIR}/src/testing-libgdf.cu
              ${binary_operation_source_files})

cuda_add_executable(testing-libgdf ${SRC_FILES})

target_link_libraries(testing-libgdf gdf Google::Flatbuffers ${BLAZINGDB_PROTOCOL_STATIC_LIB} ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY} ${CUDA_NVTX_LIBRARY})

if (NOT LIBGDF_HOME)
    target_link_libraries(testing-libgdf NVStrings)
endif()

# Print the project summary
#TODO cmake bug, it says blazingdb-protocol was not deteected vut everything seems
#fine (cmake test fail with link issues with pthreads_create)
#feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)

#END MAIN #
