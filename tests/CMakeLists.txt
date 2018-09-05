#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

#pass the dependency libraries as optional arguments using ${ARGN}
#NOTE the order of libraries matter, so try to link first with the most high level lib
function(configure_test TEST_NAME Tests_SRCS)
    message(STATUS "${TEST_NAME} will link against: gdf arrow")

    include_directories(${PROJECT_SOURCE_DIR}/src)

    set(LIBGDF_ROOT ${CMAKE_BINARY_DIR}/thirdparty/libgdf)
    set(LIBGDF_LIB_DIR ${LIBGDF_ROOT}/src/libgdf)
    link_directories(${LIBGDF_LIB_DIR})

    #target_link_libraries(${TEST_NAME} libgdf.so ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})

    set(LIBGDF_INCLUDE_DIR ${LIBGDF_ROOT}/src/libgdf/include)
    include_directories(${LIBGDF_INCLUDE_DIR})
    message(STATUS "${LIBGDF_INCLUDE_DIR} include dir")

    cuda_add_executable(${TEST_NAME} ${Tests_SRCS})
    #target_link_libraries(${TEST_NAME} gmock_main gmock GTest::GTest gdf arrow)
    target_link_libraries(${TEST_NAME} gmock_main gmock GTest::GTest gdf)
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")

    #register the target as CMake test so we can use ctest for this executable
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

## Main ##

enable_testing()

# Configure unit tests with Google Test
# -------------------
message(STATUS "******** Configuring tests ********")

add_subdirectory(foo-sample)
add_subdirectory(logical-filter)

message(STATUS "******** Tests are ready ********")
