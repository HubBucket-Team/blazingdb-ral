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

macro(CONFIGURE_ARROW_EXTERNAL_PROJECT)
    set(ARROW_ROOT ${CMAKE_BINARY_DIR}/arrow)

    #NOTE
    # libcudf.so` is now built with the old ABI `-D_GLIBCXX_USE_CXX11_ABI=0`
    # If you build Arrow from source, you can fix this by using `-DARROW_TENSORFLOW=ON`.
    # This forces Arrow to use the old ABI.
    set(ARROW_CMAKE_ARGS " -DARROW_WITH_LZ4=OFF"
                         " -DARROW_WITH_ZSTD=OFF"
                         " -DARROW_WITH_BROTLI=OFF"
                         " -DARROW_WITH_SNAPPY=OFF"
                         " -DARROW_WITH_ZLIB=OFF"
                         " -DARROW_BUILD_STATIC=ON"
                         " -DARROW_BUILD_SHARED=OFF"
                         " -DARROW_BOOST_USE_SHARED=OFF"
                         " -DARROW_BUILD_TESTS=OFF"
                         " -DARROW_TEST_MEMCHECK=OFF"
                         " -DARROW_BUILD_BENCHMARKS=OFF"
                         " -DARROW_IPC=OFF" # we don't need ipc for blazingdb-io
                         " -DARROW_COMPUTE=OFF"
                         " -DARROW_GPU=OFF"
                         " -DARROW_JEMALLOC=OFF"
                         " -DARROW_BOOST_VENDORED=OFF"
                         " -DARROW_PYTHON=OFF"
                         " -DARROW_HDFS=ON"
                         " -DARROW_TENSORFLOW=ON" # enable old ABI for C/C++
                         " -DARROW_PARQUET=OFF" # we don't need parquet for blazingdb-io
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

if (ARROW_INSTALL_DIR)
    message(STATUS "ARROW_INSTALL_DIR defined, it will use vendor version from ${ARROW_INSTALL_DIR}")
    set(ARROW_ROOT "${ARROW_INSTALL_DIR}")
else()
    message(STATUS "ARROW_INSTALL_DIR not defined, it will be built from sources")
    configure_arrow_external_project()
    set(ARROW_ROOT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-install/")
endif()

set(ENV{ARROW_HOME} ${ARROW_ROOT})

find_package(Arrow REQUIRED)
set_package_properties(Arrow PROPERTIES TYPE REQUIRED
    PURPOSE "Apache Arrow is a cross-language development platform for in-memory data."
    URL "https://arrow.apache.org")

set(ARROW_INCLUDEDIR ${ARROW_ROOT}/include/)

include_directories(${ARROW_INCLUDEDIR})

# END MAIN #
