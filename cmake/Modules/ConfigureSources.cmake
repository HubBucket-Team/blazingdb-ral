#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

function(compile_flatbuffers_schema_to_cpp SRC_FBS)
  get_filename_component(SRC_FBS_DIR ${SRC_FBS} PATH)
  string(REGEX REPLACE "\\.fbs$" "_generated.h" GEN_HEADER ${SRC_FBS})
  add_custom_command(
    OUTPUT ${GEN_HEADER}
    COMMAND "${FLATBUFFERS_FLATC_EXECUTABLE}" -c --no-includes --gen-mutable
            --gen-object-api -o "${SRC_FBS_DIR}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/srs"
            "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FBS}"
    DEPENDS flatc)
endfunction()

function(compile_flatbuffers_schema_to_binary SRC_FBS)
  get_filename_component(SRC_FBS_DIR ${SRC_FBS} PATH)
  string(REGEX REPLACE "\\.fbs$" ".bfbs" GEN_BINARY_SCHEMA ${SRC_FBS})
  add_custom_command(
    OUTPUT ${GEN_BINARY_SCHEMA}
    COMMAND "${FLATBUFFERS_FLATC_EXECUTABLE}" -b --schema -o "${SRC_FBS_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FBS}"
    DEPENDS flatc)
endfunction()

# Add *.c, *.cpp and *.cu files (sources only, not headers)
macro(CONFIGURE_SOURCES)
    compile_flatbuffers_schema_to_cpp(src/Protocol/Schema/datatypes.fbs)

    set(Protocol_SRCS
        src/Protocol/MessageBuilder.cpp
    )

    set(Simplicity_SRCS
        ${Protocol_SRCS}

        src/Simplicity.cu
    )
endmacro()
