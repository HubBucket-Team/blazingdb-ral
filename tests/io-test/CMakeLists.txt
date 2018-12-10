include_directories(
    ${CMAKE_SOURCE_DIR}/thirdparty/jitify
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper)

set(source_files
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/code/gdf-data.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/code/kernel.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/code/operation.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/code/traits.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/core/binop.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/core/launcher.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/util/operator.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/util/type.cpp
)

set(parse_csv-test_SRCS
${CMAKE_SOURCE_DIR}/src/GDFCounter.cu

  ${CMAKE_SOURCE_DIR}/src/cuDF/Allocator.cpp
  ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
  ${CMAKE_SOURCE_DIR}/src/io/data_parser/CSVParser.cpp
  ${CMAKE_SOURCE_DIR}/src/io/data_provider/UriDataProvider.cpp
  ${source_files}
  parse_csv.cu)

configure_test(parse_csv-test "${parse_csv-test_SRCS}")

