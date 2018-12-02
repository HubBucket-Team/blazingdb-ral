#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

# BEGIN macros

macro(CONFIGURE_FLATBUFFERS_EXTERNAL_PROJECT)
    # Download and unpack flatbuffers at configure time
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/FlatBuffers.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/flatbuffers-download/CMakeLists.txt)

    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/flatbuffers-download/
    )

    if(result)
        message(FATAL_ERROR "CMake step for flatbuffers failed: ${result}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . -- -j8
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/flatbuffers-download/
    )

    if(result)
        message(FATAL_ERROR "Build step for flatbuffers failed: ${result}")
    endif()
endmacro()

# END macros

# BEGIN MAIN #

if (FLATBUFFERS_HOME AND FLATBUFFERS_BUILD )
    message(STATUS "FLATBUFFERS_HOME defined, it will use vendor version from build ${FLATBUFFERS_HOME}")
    message(STATUS "FLATBUFFERS_BUILD defined, it will use vendor version from build ${FLATBUFFERS_BUILD}")
else()
    message(STATUS "FLATBUFFERS_HOME and FLATBUFFERS_BUILD are not defined, it will be built from sources")
    configure_flatbuffers_external_project()
    set(FLATBUFFERS_HOME "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/flatbuffers-install/")
    set(FLATBUFFERS_BUILD "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/flatbuffers-build/")
endif()

message(STATUS "FLATBUFFERS_HOME: " ${FLATBUFFERS_HOME})
message(STATUS "FLATBUFFERS_BUILD: " ${FLATBUFFERS_BUILD})

find_package(FlatBuffers REQUIRED)
set_package_properties(FlatBuffers
    PROPERTIES TYPE REQUIRED
    PURPOSE "FlatBuffers is an efficient cross platform serialization library."
    URL "https://google.github.io/flatbuffers/")

if (NOT FLATBUFFERS_FOUND)
    message(FATAL_ERROR "FlatBuffers not found, please check your settings.")
endif()

message(STATUS "flatbuffers installation found in ${FLATBUFFERS_HOME}")
message(STATUS "flatbuffers compiler found in ${FLATBUFFERS_BUILD}")

include_directories(${FLATBUFFERS_INCLUDEDIR})
link_directories(${FLATBUFFERS_LIBDIR})

# END MAIN #
