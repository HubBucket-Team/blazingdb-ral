#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

#pass the dependency libraries as optional arguments using ${ARGN}
#NOTE the order of libraries matter, so try to link first with the most high level lib
function(configure_benchmarks BENCHMARK_NAME Bench_SRCS)
    include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests/utils ${GBENCHMARK_INCLUDE_DIR})
    link_directories(${GBENCHMARK_LIBRARY_DIR})

    add_executable(${BENCHMARK_NAME} ${Bench_SRCS})

    # TODO percy do not use libgdf.so here and use gdf (the static lib) instead
    target_link_libraries(${BENCHMARK_NAME}
        benchmark
        benchmark_main

        ${BLAZINGDB_IO_STATIC_LIB}
        ${BLAZINGDB_PROTOCOL_STATIC_LIB}

        ${LIBGDF_STATIC_LIB}
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

    set_target_properties(${BENCHMARK_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gbenchmarks/")

endfunction()

## Main ##

# Configure benchmarks with Google Benchmark
# -------------------
message(STATUS "******** Configuring Benchmarks ********")

#add_subdirectory(utils)


message(STATUS "******** Benchmarks are ready ********")
