#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
#=============================================================================

#  github/blazingdb/blazingdb-io (libblazing_io.a)
# BLAZINGIO_ROOT hints the location
#
# This module defines
# LIB_BLAZINGIO_FOUND
# LIB_BLAZINGIO_INCLUDEDIR Preferred include directory e.g. <prefix>/include
# LIB_BLAZINGIO_INCLUDE_DIR, directory containing libblazing_io headers
# LIB_BLAZINGIO_LIBS, libblazing_io libraries
# LIB_BLAZINGIO_LIBDIR, directory containing libblazing_io libraries
# BLAZINGIO_STATIC_LIB, path to libblazing_io.a
# blazingdb-io - static library

# If BLAZINGIO_ROOT is not defined try to search in the default system path
if ("${BLAZINGIO_ROOT}" STREQUAL "")
    set(BLAZINGIO_ROOT "/usr")
endif()

set(LIB_BLAZINGIO_SEARCH_LIB_PATH
  ${BLAZINGIO_ROOT}/lib
  ${BLAZINGIO_ROOT}/build
)

set(LIB_BLAZINGIO_SEARCH_INCLUDE_DIR
  ${BLAZINGIO_ROOT}/include
)

find_path(LIB_BLAZINGIO_INCLUDE_DIR FileSystem/Path.h
    PATHS ${LIB_BLAZINGIO_SEARCH_INCLUDE_DIR}
    NO_DEFAULT_PATH
    DOC "Path to libblazing_io headers"
)

#find_library(LIB_BLAZINGIO_LIBS NAMES gdf
#    PATHS ${LIB_BLAZINGIO_SEARCH_LIB_PATH}
#    NO_DEFAULT_PATH
#    DOC "Path to libblazing_io library"
#)

#TODO percy change to libblazing_io.a once cudf supports static build
find_library(BLAZINGIO_STATIC_LIB NAMES libblazing-io-lib.a
    PATHS ${LIB_BLAZINGIO_SEARCH_LIB_PATH}
    NO_DEFAULT_PATH 
    DOC "Path to libblazing_io static library"
)

if (NOT BLAZINGIO_STATIC_LIB)
    message(FATAL_ERROR "libblazing_io includes and libraries NOT found. "
      "Looked for headers in ${LIB_BLAZINGIO_SEARCH_INCLUDE_DIR}, "
      "and for libs in ${LIB_BLAZINGIO_SEARCH_LIB_PATH}")
    set(LIB_BLAZINGIO_FOUND FALSE)
else()
    set(LIB_BLAZINGIO_INCLUDEDIR ${BLAZINGIO_ROOT}/include/)
    set(LIB_BLAZINGIO_LIBDIR ${BLAZINGIO_ROOT}/lib) # TODO percy make this part cross platform
    set(LIB_BLAZINGIO_FOUND TRUE)
    add_library(blazingdb-io STATIC IMPORTED)
    set_target_properties(blazingdb-io PROPERTIES IMPORTED_LOCATION "${BLAZINGIO_STATIC_LIB}")
endif ()

mark_as_advanced(
  LIB_BLAZINGIO_FOUND
  LIB_BLAZINGIO_INCLUDEDIR
  LIB_BLAZINGIO_INCLUDE_DIR
#   LIB_BLAZINGIO_LIBS
  BLAZINGIO_STATIC_LIB
  blazingdb-io
)