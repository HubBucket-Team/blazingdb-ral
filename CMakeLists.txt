#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

cmake_minimum_required(VERSION 2.8)
project(testing-cmake-libgdf)

set(CMAKE_CXX_STANDARD 11)
message(STATUS "Using C++ standard: c++${CMAKE_CXX_STANDARD}")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
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
#include(ConfigureArrow)
include(ConfigureFlatBuffers)
include(ConfigureGoogleTest)

# BEGIN MAIN #
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/CalciteInterpreter.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/ColumnManipulation.cu
#              ${CMAKE_CURRENT_SOURCE_DIR}/src/JoinProcessor.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/ResultSetRepository.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/LogicalFilter.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/StringUtil.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/QueryState.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/Utils.cu
              ${CMAKE_CURRENT_SOURCE_DIR}/src/GDFCounter.cu
              ${CMAKE_CURRENT_SOURCE_DIR}/src/GDFColumn.cu
              ${CMAKE_CURRENT_SOURCE_DIR}/src/testing-libgdf.cu

              ${CMAKE_CURRENT_SOURCE_DIR}/src/ipc/calcite_client.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/protocol/MessageBuilder.cpp
)

cuda_add_executable(testing-libgdf ${SRC_FILES})
target_link_libraries(testing-libgdf libgdf.a ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})

# NOTE Each time we build the target testing-libgdf we will compile the flatbuffer schemas
add_dependencies(testing-libgdf FlatBuffersSchemas)

# Print the project summary
feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)

#END MAIN #
