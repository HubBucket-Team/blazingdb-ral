#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(blazingdb-ral C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQURIED ON)

message(STATUS "Using C++ standard: c++${CMAKE_CXX_STANDARD}")

# TODO percy felipe alexander remove these lines when cudf stop using old C++ ABI
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
message(AUTHOR_WARNING "Using old C++ ABI: -D_GLIBCXX_USE_CXX11_ABI=0")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_MODULE_PATH:" "${CMAKE_MODULE_PATH}")

# Include CMake modules
include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)

# Configure CUDA
include(ConfigureCUDA)

# NOTE Do not change the inclusion order
# Configure dependencies
include(ConfigureBoost)
include(ConfigureAWSSDK)
include(ConfigureFlatBuffers)
include(ConfigureLz4)
include(ConfigureZSTD)
include(ConfigureBrotli)
include(ConfigureSnappy)
include(ConfigureThrift)
include(ConfigureArrow)
include(ConfigureParquet)
include(ConfigureLibGDF)
include(ConfigureBlazingDBProtocol)
include(ConfigureBlazingDBIO)
include(ConfigureGoogleBenchmark)

find_package(Threads)

# BEGIN MAIN #
include_directories(
    ${LIB_BLAZINGIO_INCLUDEDIR}
    ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/thirdparty/jitify
    ${NVSTRINGS_INCLUDEDIR}
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper)

## cuDF adapter source files
add_subdirectory(${CMAKE_SOURCE_DIR}/src/cuDF)
get_directory_property(cudf_adapter_source_files DIRECTORY ${CMAKE_SOURCE_DIR}/src/cuDF DEFINITION source_files)

## ParquetReader source files
add_subdirectory(${CMAKE_SOURCE_DIR}/src/cuio/parquet)
get_directory_property(parquet_reader_source_files DIRECTORY ${CMAKE_SOURCE_DIR}/src/cuio/parquet DEFINITION source_files)

## Target source files
set(SRC_FILES ${CMAKE_SOURCE_DIR}/src/io/data_provider/UriDataProvider.cpp
              ${CMAKE_SOURCE_DIR}/src/FreeMemory.cpp
              ${CMAKE_SOURCE_DIR}/src/io/data_parser/ParquetParser.cpp
              ${CMAKE_SOURCE_DIR}/src/io/data_parser/CSVParser.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/Config/Config.cpp
              ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
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
              ${parquet_reader_source_files})

add_library(testing-libgdf_lib ${SRC_FILES})

target_link_libraries(testing-libgdf_lib
    ${BLAZINGDB_IO_STATIC_LIB}
    Threads::Threads
    ${ZEROMQ_STATIC_LIB}
    ${BLAZINGDB_PROTOCOL_STATIC_LIB}

    ${LIBGDF_STATIC_LIB}
    NVCategory
    NVStrings
    rmm

    cudart
    cuda
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_NVRTC_LIBRARY}
    ${CUDA_NVTX_LIBRARY}

    ${PARQUET_STATIC_LIB}
    ${ARROW_STATIC_LIB}
    ${THRIFT_STATIC_LIB}
    ${SNAPPY_STATIC_LIB}
    ${BROTLI_STATIC_LIB}
    ${ZSTD_STATIC_LIB}
    ${LZ4_STATIC_LIB}
    ${FLATBUFFERS_STATIC_LIB}

    aws-cpp-sdk-s3
    aws-cpp-sdk-s3-encryption

    Boost::filesystem
    Boost::system
    Boost::regex
)

set(SRC_FILES_APP ${CMAKE_SOURCE_DIR}/src/testing-libgdf.cu)
add_executable(testing-libgdf ${SRC_FILES_APP})
target_link_libraries(testing-libgdf testing-libgdf_lib)

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
#TODO cmake bug, it says blazingdb-protocol was not deteected but everything seems
#fine (cmake test fail with link issues with pthreads_create)
#feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)

#END MAIN #
