#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# BEGIN macros

macro(CONFIGURE_PARQUET_EXTERNAL_PROJECT)
    set(ENV{FLATBUFFERS_HOME} ${FLATBUFFERS_INSTALL_DIR})

    #NOTE
    # libcudf.so` is now built with the old ABI `-D_GLIBCXX_USE_CXX11_ABI=0`
    # If you build Arrow from source, you can fix this by using `-DPARQUET_TENSORFLOW=ON`.
    # This forces Arrow to use the old ABI.
    set(PARQUET_CMAKE_ARGS " -DPARQUET_WITH_LZ4=ON"
                         " -DPARQUET_WITH_ZSTD=ON"
                         " -DPARQUET_WITH_BROTLI=ON"
                         " -DPARQUET_WITH_SNAPPY=ON"
                         " -DPARQUET_WITH_ZLIB=ON"
                         " -DPARQUET_BUILD_STATIC=ON"
                         " -DPARQUET_BUILD_SHARED=OFF"
                         " -DPARQUET_BOOST_USE_SHARED=OFF"
                         " -DPARQUET_BUILD_TESTS=OFF"
                         " -DPARQUET_TEST_MEMCHECK=OFF"
                         " -DPARQUET_BUILD_BENCHMARKS=OFF"
                         " -DPARQUET_IPC=ON" # need ipc for blazingdb-ral (because cudf)
                         " -DPARQUET_COMPUTE=OFF"
                         " -DPARQUET_GPU=OFF"
                         " -DPARQUET_JEMALLOC=OFF"
                         " -DPARQUET_BOOST_VENDORED=OFF"
                         " -DPARQUET_PYTHON=OFF"
                         " -DPARQUET_HDFS=ON" # disable when blazingdb-io don't use arrow for hdfs
                         " -DPARQUET_TENSORFLOW=ON" # enable old ABI for C/C++
                         " -DPARQUET_PARQUET=ON"
    )

    # Download and unpack arrow at configure time
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/Arrow.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-download/CMakeLists.txt)

    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-download/
    )

    if(result)
        message(FATAL_ERROR "CMake step for arrow failed: ${result}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . -- -j8
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-download/
    )

    if(result)
        message(FATAL_ERROR "Build step for arrow failed: ${result}")
    endif()
endmacro()

# END macros

# BEGIN MAIN #

# NOTE since parquet and arrow are in the same repo is safe to pass the arrrow installation dir here
set(PARQUET_INSTALL_DIR ${ARROW_INSTALL_DIR})

if (PARQUET_INSTALL_DIR)
    message(STATUS "PARQUET_INSTALL_DIR defined, it will use vendor version from ${PARQUET_INSTALL_DIR}")
    set(PARQUET_ROOT "${PARQUET_INSTALL_DIR}")
else()
    message(STATUS "PARQUET_INSTALL_DIR not defined, it will be built from sources")
    configure_parquet_external_project()
    set(PARQUET_ROOT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-install/")
endif()

set(ENV{PARQUET_HOME} ${PARQUET_ROOT})

find_package(Parquet REQUIRED)
set_package_properties(Parquet PROPERTIES TYPE REQUIRED
    PURPOSE "Apache Parquet CPP is a C++ library to read and write the Apache Parquet columnar data format."
    URL "https://arrow.apache.org")

set(PARQUET_INCLUDEDIR ${PARQUET_ROOT}/include/)

include_directories(${PARQUET_INCLUDEDIR})
link_directories(${PARQUET_ROOT}/lib/)

# END MAIN #
