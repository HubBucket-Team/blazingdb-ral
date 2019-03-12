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
    ${CMAKE_SOURCE_DIR}/src/Config/Config.cpp
)

set(common_source_files
    ${CMAKE_SOURCE_DIR}/src/Utils.cu
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    ${CMAKE_SOURCE_DIR}/src/FreeMemory.cpp
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${CMAKE_SOURCE_DIR}/src/ColumnManipulation.cu
    ${CMAKE_SOURCE_DIR}/src/CodeTimer.cpp
    ${CMAKE_SOURCE_DIR}/src/JoinProcessor.cpp
    ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
    ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
    ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
    ${CMAKE_SOURCE_DIR}/src/Interpreter/interpreter_cpp.cu
    ${CMAKE_SOURCE_DIR}/src/CalciteInterpreter.cpp
    ${CMAKE_SOURCE_DIR}/src/communication/CommunicationData.cpp
    ${CMAKE_SOURCE_DIR}/src/communication/adapter/GpuFunctionsAdapter.cpp
    ${CMAKE_SOURCE_DIR}/src/communication/factory/MessageFactory.cpp
    ${CMAKE_SOURCE_DIR}/src/communication/network/Client.cpp
    ${CMAKE_SOURCE_DIR}/src/communication/network/Server.cpp
    ${CMAKE_SOURCE_DIR}/src/cuDF/Allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/cuDF/generator/sample_generator.cu
    ${CMAKE_SOURCE_DIR}/src/distribution/Exception.cpp
    ${CMAKE_SOURCE_DIR}/src/distribution/internal.cpp
    ${CMAKE_SOURCE_DIR}/src/distribution/NodeColumns.cpp
    ${CMAKE_SOURCE_DIR}/src/distribution/NodeSamples.cpp
    ${CMAKE_SOURCE_DIR}/src/distribution/partitions.cpp
    ${CMAKE_SOURCE_DIR}/src/distribution/primitives.cpp
    ${CMAKE_SOURCE_DIR}/src/exception/RalException.cpp
    ${CMAKE_SOURCE_DIR}/src/Traits/RuntimeTraits.cpp
    ${CMAKE_SOURCE_DIR}/src/operators/OrderBy.cpp
    ${source_files}
)

set(calcite_interpreter_test_SRCS
    ${common_source_files}
    calcite_interpreter-test.cu
)

set(calcite_interpreter_where_test_SRCS
    ${common_source_files}
    calcite_interpreter-where-test.cu
)

configure_test(calcite-interpreter-test "${calcite_interpreter_test_SRCS}")
configure_test(calcite-interpreter-where-test "${calcite_interpreter_where_test_SRCS}")

target_link_libraries(calcite-interpreter-test
                      ${cuDF_adapter_testing_library}
                      ${cuDF_adapter_testing_dependencies}
)
