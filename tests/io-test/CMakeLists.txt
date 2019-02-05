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
    #todo remove when the PR https://github.com/rapidsai/cudf/pull/643 is accepted 
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/replace/replace.cu
)


set(parse_csv-test_SRCS
  ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
  ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
  ${CMAKE_SOURCE_DIR}/src/FreeMemory.cpp
  ${CMAKE_SOURCE_DIR}/src/cuDF/Allocator.cpp
  ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
  ${CMAKE_SOURCE_DIR}/src/io/data_parser/CSVParser.cpp
  ${CMAKE_SOURCE_DIR}/src/io/data_provider/UriDataProvider.cpp
  ${source_files}
  parse_csv.cu
)
  
set(parse_parquet-test_SRCS
    ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    ${CMAKE_SOURCE_DIR}/src/FreeMemory.cpp
    ${CMAKE_SOURCE_DIR}/src/cuDF/Allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${CMAKE_SOURCE_DIR}/src/io/data_parser/ParquetParser.cpp
    ${CMAKE_SOURCE_DIR}/src/io/data_provider/UriDataProvider.cpp
    ${parquet_reader_source_files}
    ${source_files}
    parse_parquet.cu
)
 
configure_test(parse_csv-test "${parse_csv-test_SRCS}")
configure_test(parse_parquet-test "${parse_parquet-test_SRCS}") 
