#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

#pass the dependency libraries as optional arguments using ${ARGN}
#NOTE the order of libraries matter, so try to link first with the most high level lib
function(configure_test TEST_NAME Tests_SRCS)
    message(STATUS "${TEST_NAME} will link against: gdf arrow")

    include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests/utils)

    cuda_add_executable(${TEST_NAME} ${Tests_SRCS})

    # TODO percy do not use libgdf.so here and use gdf (the static lib) instead
    target_link_libraries(${TEST_NAME} gmock_main gmock GTest::GTest ${LIBGDF_STATIC_LIB} ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY} ${CUDA_NVTX_LIBRARY})
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")

    #register the target as CMake test so we can use ctest for this executable
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

## Main ##

enable_testing()


## Helper libraries for TESTING
# Static library (libcuDF_adapter_testing.a) - cuDF folder
include(${CMAKE_SOURCE_DIR}/src/cuDF/CMakeLists.txt)
include(ConfigureArrow)
include(ConfigureBlazingIO)

message(STATUS "static lib is ${BLAZINGIO_STATIC_LIB}")
set(cuDF_adapter_testing_files ${source_files})
set(cuDF_adapter_testing_library "cuDF_adapter_testing")
set(cuDF_adapter_testing_dependencies "rmm ${BLAZINGIO_STATIC_LIB}")

add_library(${cuDF_adapter_testing_library} STATIC ${cuDF_adapter_testing_files})

target_include_directories(${cuDF_adapter_testing_library} PRIVATE ${CMAKE_SOURCE_DIR}/src/)


# Configure unit tests with Google Test
# -------------------
message(STATUS "******** Configuring tests ********")

add_subdirectory(utils)

add_subdirectory(logical-filter)
add_subdirectory(calcite-interpreter)
add_subdirectory(integration)
add_subdirectory(dtypes)
add_subdirectory(pjoins)
add_subdirectory(evaluate_query)
add_subdirectory(order-by)
add_subdirectory(gdf-ref-counter)
add_subdirectory(where)
add_subdirectory(joins)
add_subdirectory(tpch)

message(STATUS "******** Tests are ready ********")
