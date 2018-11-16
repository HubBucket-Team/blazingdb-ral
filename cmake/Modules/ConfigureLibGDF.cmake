#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

# BEGIN macros

macro(CONFIGURE_GPU_LIBGDF_EXTERNAL_PROJECT)
    set(ENV{NVSTRINGS_ROOT} ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-download/nvstrings-prefix/src/nvstrings/)

    # Download and unpack libgdf at configure time
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/LibGDF.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-download/CMakeLists.txt)

    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-download/
    )

    if(result)
        message(FATAL_ERROR "CMake step for libgdf failed: ${result}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . -- -j8
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-download/
    )

    if(result)
        message(FATAL_ERROR "Build step for libgdf failed: ${result}")
    endif()
endmacro()

# END macros

# BEGIN MAIN #

if (LIBGDF_HOME)
    if (NOT NVSTRINGS_HOME)
        message(FATAL_ERROR "If you use the LIBGDF_HOME argument then you need pass the NVSTRINGS_HOME argument too (the home installation of nvstrings)")
    endif()

    message(STATUS "LIBGDF_HOME defined, it will use vendor version from ${LIBGDF_HOME}")
    set(LIBGDF_ROOT "${LIBGDF_HOME}")
else()
    message(STATUS "LIBGDF_HOME not defined, it will be built from sources")
    configure_gpu_libgdf_external_project()
    set(LIBGDF_ROOT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-install/")
endif()

set(NVSTRINGS_LIBDIR ${NVSTRINGS_HOME}/lib/)
link_directories(${NVSTRINGS_LIBDIR})

find_package(LibGDF REQUIRED)
set_package_properties(LibGDF PROPERTIES TYPE REQUIRED
    PURPOSE "libgdf is a C library for implementing common functionality for a GPU Data Frame."
    URL "https://github.com/gpuopenanalytics/libgdf")

if(NOT LIBGDF_FOUND)
    message(FATAL_ERROR "libgdf not found, please check your settings.")
endif()

message(STATUS "libgdf found in ${LIBGDF_ROOT}")
include_directories(${LIBGDF_INCLUDEDIR})
# TODO percy seems cmake bug: we cannot define target dirs per cuda target
# ... see if works in future cmake versions
link_directories(${LIBGDF_LIBDIR})

# END MAIN #

