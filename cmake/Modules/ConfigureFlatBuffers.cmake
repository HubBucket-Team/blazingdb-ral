#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

# Locate the FlatBuffers package.
# Requires that you build with:
#   -DFLATBUFFERS_HOME:PATH=/path/to/flatbuffers_install_dir
message(STATUS "FLATBUFFERS_HOME: " ${FLATBUFFERS_HOME})
find_package(FlatBuffers REQUIRED)
set_package_properties(FlatBuffers PROPERTIES TYPE REQUIRED PURPOSE "FlatBuffers is an efficient cross platform serialization library." URL "https://google.github.io/flatbuffers/")

if(NOT FLATBUFFERS_FOUND)
    message(FATAL_ERROR "FlatBuffers not found, please check your settings.")
endif()


include_directories(${FLATBUFFERS_INCLUDE_DIR})
include_directories(${FLATBUFFERS_ROOT}/include/)
link_directories(${FLATBUFFERS_ROOT}/lib/)
