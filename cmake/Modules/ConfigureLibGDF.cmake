#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

if (DEFINED ENV{LIBGDF_HOME})
    set(LIBGDF_HOME_HOME "$ENV{LIBGDF_HOME}")
else()
    message(STATUS "LIBGDF_HOME not defined, it will be built from sources")
endif()

# BEGIN macros

macro(CONFIGURE_GPU_LIBGDF_EXTERNAL_PROJECT)
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
        COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-download/
    )

    if(result)
        message(FATAL_ERROR "Build step for libgdf failed: ${result}")
    endif()
endmacro()

# define arrow, flatbuffer, etc. dirs
macro(CONFIGURE_LIBGDF_TRANSITIVE_DEPENDENCIES)
    # Add transitive dependency: Apache Arrow
    #set(FLATBUFFERS_ROOT ${ARROW_DOWNLOAD_BINARY_DIR}/arrow-prefix/src/arrow-build/flatbuffers_ep-prefix/src/flatbuffers_ep-install/)
    #set(FLATBUFFERS_HOME ${FLATBUFFERS_ROOT})

    # Add transitive dependency: Flatbuffers
    if (DEFINED ENV{LIBGDF_HOME})
        set(FLATBUFFERS_ROOT ${LIBGDF_HOME_HOME}/build/${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-download/arrow-prefix/src/arrow-build/flatbuffers_ep-prefix/src/flatbuffers_ep-install/)
    else()
        set(FLATBUFFERS_ROOT ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-build/${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-download/arrow-prefix/src/arrow-build/flatbuffers_ep-prefix/src/flatbuffers_ep-install/)
    endif()

    set(FLATBUFFERS_HOME ${FLATBUFFERS_ROOT})
endmacro()

# END macros

# BEGIN MAIN #

# TODO percy use vendored option too
#
# if("${SNAPPY_HOME}" STREQUAL "")
#     CONFIGURE_GPU_LIBGDF_EXTERNAL_PROJECT
# else()
#     find_package(Snappy REQUIRED)
# endif()

if (NOT DEFINED ENV{LIBGDF_HOME})
    configure_gpu_libgdf_external_project()
endif()

configure_libgdf_transitive_dependencies()

if (DEFINED ENV{LIBGDF_HOME})
    set(LIBGDF_ROOT "${LIBGDF_HOME_HOME}")
else()
    set(LIBGDF_ROOT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-install/")
endif()

find_package(LibGDF REQUIRED)
set_package_properties(LibGDF PROPERTIES TYPE REQUIRED
    PURPOSE "libgdf is a C library for implementing common functionality for a GPU Data Frame."
    URL "https://github.com/gpuopenanalytics/libgdf")

if(NOT LIBGDF_FOUND)
    message(AUTHOR_WARNING "libgdf not found, please check your settings.")
endif()

message(STATUS "libgdf found in ${LIBGDF_ROOT}")
include_directories(${LIBGDF_INCLUDEDIR})
# TODO percy seems cmake bug: we cannot define target dirs per cuda target
# ... see if works in future cmake versions

link_directories(${LIBGDF_LIBDIR})

# END MAIN #
