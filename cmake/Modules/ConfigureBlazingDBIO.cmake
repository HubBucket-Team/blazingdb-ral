#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

# BEGIN macros

macro(CONFIGURE_BLAZINGDB_IO_EXTERNAL_PROJECT)
    set(BLAZINGDB_IO_CMAKE_ARGS  " -DARROW_INSTALL_DIR=${ARROW_INSTALL_DIR}")

    # Download and unpack blazingdb-io at configure time
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/BlazingDBIO.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/blazingdb-io-download/CMakeLists.txt)

    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/blazingdb-io-download/
    )

    if(result)
        message(FATAL_ERROR "CMake step for blazingdb-io failed: ${result}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . -- -j8
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/blazingdb-io-download/
    )

    if(result)
        message(FATAL_ERROR "Build step for blazingdb-io failed: ${result}")
    endif()
endmacro()

# END macros

# BEGIN MAIN #

if (BLAZINGDB_IO_INSTALL_DIR)
    message(STATUS "BLAZINGDB_IO_INSTALL_DIR defined, it will use vendor version from ${BLAZINGDB_IO_INSTALL_DIR}")
    set(BLAZINGDB_IO_ROOT "${BLAZINGDB_IO_INSTALL_DIR}")
else()
    message(STATUS "BLAZINGDB_IO_INSTALL_DIR not defined, it will be built from sources")
    configure_blazingdb_io_external_project()
    set(BLAZINGDB_IO_ROOT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/blazingdb-io-install/")
endif()

find_package(BlazingDBIO REQUIRED)
set_package_properties(BlazingDBIO PROPERTIES TYPE REQUIRED
    PURPOSE "BlazingDBIO has the file system and I/O definitions for BlazingSQL."
    URL "https://github.com/BlazingDB/blazingdb-io")

if(NOT BLAZINGDB_IO_FOUND)
    message(FATAL_ERROR "blazingdb-io not found, please check your settings.")
endif()

message(STATUS "blazingdb-io found in ${BLAZINGDB_IO_ROOT}")
include_directories(${BLAZINGDB_IO_INCLUDEDIR})

link_directories(${BLAZINGDB_IO_LIBDIR})

# END MAIN #
