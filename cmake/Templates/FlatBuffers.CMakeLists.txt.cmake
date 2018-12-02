#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

cmake_minimum_required(VERSION 2.8.12)

cmake_policy(SET CMP0048 NEW)

project(libgdf-download NONE)

include(ExternalProject)

set(FLATBUFFERS_VERSION 02a7807dd8d26f5668ffbbec0360dc107bbfabd5)

message(STATUS "Downloading flatbuffers version: ${FLATBUFFERS_VERSION}")

ExternalProject_Add(flatbuffers
    GIT_REPOSITORY    https://github.com/google/flatbuffers.git
    GIT_TAG           ${FLATBUFFERS_VERSION}
    SOURCE_DIR        "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/flatbuffers-src"
    BINARY_DIR        "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/flatbuffers-build"
    INSTALL_DIR       "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/flatbuffers-install"
    CMAKE_ARGS        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/flatbuffers-install
    UPDATE_COMMAND ""
)
