#=============================================================================
# Copyright 2018-2019 BlazingDB, Inc.
#     Copyright 2018-2019 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(blazingdb-ral C CXX CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# NOTE Import cmake modules from blazingdb-dependencies (Mandatory: First step!)
include(ImportCMakeModules)
include(ConfigureBlazingDBHostCompiler)

# Include CMake modules
include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)

# NOTE Configure dependencies (always include this one first)
include(ConfigureBlazingDBDependencies)

# Configure CUDA
include(ConfigureCUDA)

# NOTE Do not change the inclusion order
#include(ConfigureBoost)
include(ConfigureAWSSDK)
#include(ConfigureFlatBuffers)
#include(ConfigureLz4)
#include(ConfigureZSTD)
#include(ConfigureBrotli)
#include(ConfigureSnappy)
#include(ConfigureThrift)
#include(ConfigureArrow)
#include(ConfigureParquet)
#include(ConfigureRMM)
#include(ConfigureNVStrings)
#include(ConfigureJitify)
include(ConfigureLibGDF)
include(ConfigureBlazingDBProtocol)
include(ConfigureBlazingDBCommunication)
include(ConfigureBlazingDBIO)
include(ConfigureGoogleBenchmark)
#include(ConfigureRapidJSON)

find_package(Threads)

# BEGIN MAIN #

# TODO percy move this to tool-chain and add support for cuda def and more than 1 defs
add_definitions(${CXX_DEFINES})
get_directory_property(CXX_COMPILE_DEFINITIONS DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
message(STATUS "C++ compiler definitions: ${CXX_COMPILE_DEFINITIONS}")

include_directories(
    ${LIB_BLAZINGIO_INCLUDEDIR}
    ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/thirdparty/jitify
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper
    ${RAPIDJSON_INCLUDE_DIRS}
    ${NVSTRINGS_INCLUDEDIR}
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper)

## cuDF adapter source files
add_subdirectory(${CMAKE_SOURCE_DIR}/src/cuDF)
get_directory_property(cudf_adapter_source_files DIRECTORY ${CMAKE_SOURCE_DIR}/src/cuDF DEFINITION source_files)

## communication primitives
add_subdirectory(${CMAKE_SOURCE_DIR}/src/communication)
get_directory_property(communication_source_files DIRECTORY ${CMAKE_SOURCE_DIR}/src/communication DEFINITION source_files)

## communication primitives
add_subdirectory(${CMAKE_SOURCE_DIR}/src/distribution)
get_directory_property(distribution_source_files DIRECTORY ${CMAKE_SOURCE_DIR}/src/distribution DEFINITION source_files)

## Target source files
set(SRC_FILES ${CMAKE_SOURCE_DIR}/src/config/BlazingConfig.cpp
              ${CMAKE_SOURCE_DIR}/src/config/GPUManager.cu
              ${CMAKE_SOURCE_DIR}/src/exception/RalException.cpp
              ${CMAKE_SOURCE_DIR}/src/operators/OrderBy.cpp
              ${CMAKE_SOURCE_DIR}/src/operators/JoinOperator.cpp
              ${CMAKE_SOURCE_DIR}/src/operators/GroupBy.cpp
              ${CMAKE_SOURCE_DIR}/src/io/data_provider/UriDataProvider.cpp
              ${CMAKE_SOURCE_DIR}/src/FreeMemory.cpp
              ${CMAKE_SOURCE_DIR}/src/io/Schema.cpp
              ${CMAKE_SOURCE_DIR}/src/io/data_parser/ParquetParser.cpp
              ${CMAKE_SOURCE_DIR}/src/io/data_parser/CSVParser.cpp
              ${CMAKE_SOURCE_DIR}/src/io/data_parser/GDFParser.cpp
              ${CMAKE_SOURCE_DIR}/src/io/data_parser/ParserUtil.cpp
              ${CMAKE_SOURCE_DIR}/src/Traits/RuntimeTraits.cpp
              ${CMAKE_SOURCE_DIR}/src/utilities/RalColumn.cpp
              ${CMAKE_SOURCE_DIR}/src/utilities/TableWrapper.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/Config/Config.cpp
              ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
              ${CMAKE_SOURCE_DIR}/src/io/DataLoader.cpp
              ${CMAKE_SOURCE_DIR}/src/Interpreter/interpreter_cpp.cu
              ${CMAKE_SOURCE_DIR}/src/CalciteInterpreter.cpp
              ${CMAKE_SOURCE_DIR}/src/ColumnManipulation.cu
              ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
              ${CMAKE_SOURCE_DIR}/src/JoinProcessor.cpp
              ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
              ${CMAKE_SOURCE_DIR}/src/QueryState.cpp
              ${CMAKE_SOURCE_DIR}/src/CodeTimer.cpp
              ${CMAKE_SOURCE_DIR}/src/Utils.cu
              ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
              ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
              ${cudf_adapter_source_files}
              ${binary_operation_source_files}
              ${communication_source_files}
              ${distribution_source_files}
)

add_library(testing-libgdf_lib ${SRC_FILES})

target_link_libraries(testing-libgdf_lib
    ${BLAZINGDB_IO_STATIC_LIB}
    Threads::Threads
    ${ZEROMQ_STATIC_LIB}
    ${BLAZINGDB_PROTOCOL_STATIC_LIB}
    ${BLAZINGDB_COMMUNICATION_STATIC_LIB}

    ${LIBGDF_STATIC_LIB}
    NVCategory
    NVStrings
    rmm

    cudart
    cuda
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_NVRTC_LIBRARY}
    ${CUDA_NVTX_LIBRARY}


    aws-cpp-sdk-s3
    aws-cpp-sdk-s3-encryption


    # Library needed for blazing communication repository (debug)
    gcov
)
find_package(Boost COMPONENTS regex system filesystem REQUIRED)
target_link_libraries(testing-libgdf_lib ${Boost_LIBRARIES})
include_directories(${Boost_INCLUDE_DIRS})

find_path(RMM_INCLUDE "rmm"
          HINTS "$ENV{RMM_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include/rmm"
                "$ENV{CONDA_PREFIX}/include")

find_library(RMM_LIBRARY "rmm"
             HINTS "$ENV{RMM_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

message(STATUS "RMM: RMM_LIBRARY set to ${RMM_LIBRARY}")
message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

add_library(rmm SHARED IMPORTED ${RMM_LIBRARY})
if (RMM_INCLUDE AND RMM_LIBRARY)
    set_target_properties(rmm PROPERTIES IMPORTED_LOCATION ${RMM_LIBRARY})
endif (RMM_INCLUDE AND RMM_LIBRARY)

#################################################
# - Flatbuffers

find_path(FLATBUFERS_INCLUDE "flatbuffers"
          HINTS "$ENV{FLATBUFFERS_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include/flatbuffers"
                "$ENV{CONDA_PREFIX}/include")

find_library(NVSTRINGS_LIBRARY "flatbuffers"
          HINTS "$ENV{FLATBUFFERS_ROOT}/lib"
                "$ENV{CONDA_PREFIX}/lib/flatbuffers"
                "$ENV{CONDA_PREFIX}/lib")

add_library(flatbuffers STATIC IMPORTED ${FLATBUFFERS_LIBRARY})
###################################################################################################
# - NVStrings -------------------------------------------------------------------------------------

find_path(NVSTRINGS_INCLUDE "nvstrings"
          HINTS "$ENV{NVSTRINGS_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include/nvstrings"
                "$ENV{CONDA_PREFIX}/include")

find_library(NVSTRINGS_LIBRARY "NVStrings"
             HINTS "$ENV{NVSTRINGS_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

find_library(NVCATEGORY_LIBRARY "NVCategory"
             HINTS "$ENV{NVSTRINGS_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

message(STATUS "NVSTRINGS: NVSTRINGS_INCLUDE set to ${NVSTRINGS_INCLUDE}")
message(STATUS "NVSTRINGS: NVSTRINGS_LIBRARY set to ${NVSTRINGS_LIBRARY}")
message(STATUS "NVSTRINGS: NVCATEGORY_LIBRARY set to ${NVCATEGORY_LIBRARY}")

add_library(NVStrings SHARED IMPORTED ${NVSTRINGS_LIBRARY})
if (NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY)
    set_target_properties(NVStrings PROPERTIES IMPORTED_LOCATION ${NVSTRINGS_LIBRARY})
endif (NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY)

add_library(NVCategory SHARED IMPORTED ${NVCATEGORY_LIBRARY})
if (NVSTRINGS_INCLUDE AND NVCATEGORY_LIBRARY)
    set_target_properties(NVCategory PROPERTIES IMPORTED_LOCATION ${NVCATEGORY_LIBRARY})
endif (NVSTRINGS_INCLUDE AND NVCATEGORY_LIBRARY)


set(SRC_FILES_APP
    ${CMAKE_SOURCE_DIR}/src/testing-libgdf.cu)

add_executable(testing-libgdf ${SRC_FILES_APP})
target_link_libraries(testing-libgdf testing-libgdf_lib ${BLAZINGDB_PROTOCOL_STATIC_LIB})

# Tests
if(BUILD_TESTING)
    include(ConfigureGoogleTest)

    if(GTEST_FOUND)
        add_subdirectory(tests)
    else()
        message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
    endif()
endif()

#Benchmarks
if(GBENCHMARK_FOUND)
    add_subdirectory(benchmarks)
else()
    message(AUTHOR_WARNING "Google C++ Benchmarking Framework (Google Benchmark) not found")
endif()

# Print the project summary
feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)

#END MAIN #
