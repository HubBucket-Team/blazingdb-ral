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

    ${CMAKE_SOURCE_DIR}/src/CalciteInterpreter.cpp
    ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
    ${CMAKE_SOURCE_DIR}/src/ColumnManipulation.cu
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${CMAKE_SOURCE_DIR}/src/JoinProcessor.cpp
    ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
    ${CMAKE_SOURCE_DIR}/src/CodeTimer.cpp
    ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
    ${CMAKE_SOURCE_DIR}/src/cuDF/Allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/Utils.cu
    tpch-tests.cu 
)


set(tpch_csv_tests_src
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

    ${CMAKE_SOURCE_DIR}/src/CalciteInterpreter.cpp
    ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
    ${CMAKE_SOURCE_DIR}/src/ColumnManipulation.cu
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${CMAKE_SOURCE_DIR}/src/JoinProcessor.cpp
    ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
    ${CMAKE_SOURCE_DIR}/src/CodeTimer.cpp
    ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
    ${CMAKE_SOURCE_DIR}/src/cuDF/Allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/Utils.cu

    tpch_csv_tests.cu
)

configure_test(tpch-test "${source_files}")

configure_test(tpch_csv_tests "${tpch_csv_tests_src}")
