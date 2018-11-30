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

set(logical_filter_test_SRCS
    ${CMAKE_SOURCE_DIR}/src/Utils.cu
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
    ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
    ${source_files}
    logical_filter-test.cu
)

configure_test(logical-filter-test "${logical_filter_test_SRCS}")

target_link_libraries(logical-filter-test
                      ${cuDF_adapter_testing_library}
                      ${cuDF_adapter_testing_dependencies}
)
