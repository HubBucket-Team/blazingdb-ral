#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
#=============================================================================

# BEGIN macros

macro(CONFIGURE_LIB_BLAZINGIO_EXTERNAL_PROJECT)
 
    if(result)
        message(FATAL_ERROR "Build step for blazingdb-io failed: ${result}")
    endif()
endmacro()

# END macros

# BEGIN MAIN #

if (BLAZINGIO_HOME)
    message(STATUS "BLAZINGIO_HOME defined, it will use vendor version from ${BLAZINGIO_HOME}")
    set(BLAZINGIO_ROOT "${BLAZINGIO_HOME}")
else()
    message(STATUS "BLAZINGIO_HOME not defined, it will be built from sources")
    CONFIGURE_LIB_BLAZINGIO_EXTERNAL_PROJECT()
endif()

find_package(BlazingIO REQUIRED)
set_package_properties(blazingdb-io PROPERTIES TYPE REQUIRED
    PURPOSE "blazingdb-io is a Cpp library for implementing System File IO."
    URL "")

if(NOT BLAZINGIO_HOME)
    message(FATAL_ERROR "blazingdb-io not found, please check your settings.")
endif()

message(STATUS "blazingdb-io found in ${BLAZINGIO_ROOT} , ${LIB_BLAZINGIO_INCLUDEDIR}, ${LIB_BLAZINGIO_LIBDIR}")

include_directories(${LIB_BLAZINGIO_INCLUDEDIR} ${BLAZINGDB_PROTOCOL_INCLUDE_DIR})


# TODO percy seems cmake bug: we cannot define target dirs per cuda target
# ... see if works in future cmake versions
link_directories("${LIB_BLAZINGIO_LIBDIR}")

# END MAIN #

