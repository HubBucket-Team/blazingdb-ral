cmake_minimum_required(VERSION 2.8)
project(testing-cmake-libgdf)
set(CMAKE_CXX_STANDARD 11)
message(STATUS "Using C++ standard: c++${CMAKE_CXX_STANDARD}")

find_package(CUDA REQUIRED)

if(CUDA_FOUND)
    message(STATUS "CUDA ${CUDA_VERSION} found in ${CUDA_TOOLKIT_ROOT_DIR}")

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;--expt-extended-lambda)
    message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
else()
    message(FATAL_ERROR "CUDA not found, please check your settings.")
endif()

set(PROJECT_SRC "${PROJECT_SOURCE_DIR}/src")

set(SRC_FILES ${PROJECT_SRC}/CalciteExpressionParsing.cpp
              ${PROJECT_SRC}/CalciteInterpreter.cpp
              ${PROJECT_SRC}/ColumnManipulation.cu
              ${PROJECT_SRC}/JoinProcessor.cpp
              ${PROJECT_SRC}/LogicalFilter.cpp
              ${PROJECT_SRC}/QueryState.cpp
              ${PROJECT_SRC}/testing-libgdf.cu
)

include(ExternalProject)

set(LIBGDF_ROOT ${CMAKE_BINARY_DIR}/thirdparty/libgdf)
set(LIBGDF_LIB_DIR ${LIBGDF_ROOT}/src/libgdf)
set(LIBGDF_INCLUDE_DIR ${LIBGDF_ROOT}/src/libgdf/include)

ExternalProject_Add(libgdf_ep
	CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE 
	PREFIX ${LIBGDF_ROOT}
	DOWNLOAD_COMMAND git clone --recursive https://github.com/BlazingDB/libgdf.git
	UPDATE_COMMAND ""
	PATCH_COMMAND ""
	BINARY_DIR ${LIBGDF_ROOT}/src/libgdf
	SOURCE_DIR ${LIBGDF_ROOT}/src/libgdf
	INSTALL_DIR ${LIBGDF_ROOT}/bin
	CONFIGURE_COMMAND git checkout binary-operators-master && cmake -DCMAKE_BUILD_TYPE=Release -DBINARY_OPERATION_VERSION:STRING=V2 .
	BUILD_COMMAND make -j8
	INSTALL_COMMAND ""
	BUILD_BYPRODUCTS ${LIBGDF_LIB_DIR}/libgdf.so
)

set(PROJECT_LINK_LIBS libgdf.so)
link_directories(${LIBGDF_LIB_DIR})

include_directories(${LIBGDF_INCLUDE_DIR})

add_library(libgdf SHARED IMPORTED)
set_target_properties(libgdf PROPERTIES IMPORTED_LOCATION ${LIBGDF_LIB_DIR}/libgdf.so)
add_dependencies(libgdf libgdf_ep)

cuda_add_executable(testing-libgdf ${SRC_FILES})
target_link_libraries(testing-libgdf libgdf cuda nvrtc)
