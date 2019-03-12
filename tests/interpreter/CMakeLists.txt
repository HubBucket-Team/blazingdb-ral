include_directories(
    ${CMAKE_SOURCE_DIR}/thirdparty/jitify
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper)
 

set(source_files_src
    ${CMAKE_SOURCE_DIR}/src/Interpreter/interpreter_cpp.cu
    ${CMAKE_SOURCE_DIR}/src/CalciteInterpreter.cpp
    ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
    ${CMAKE_SOURCE_DIR}/src/ColumnManipulation.cu
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    ${CMAKE_SOURCE_DIR}/src/FreeMemory.cpp
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${CMAKE_SOURCE_DIR}/src/JoinProcessor.cpp
    ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
    ${CMAKE_SOURCE_DIR}/src/CodeTimer.cpp
    ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
    ${CMAKE_SOURCE_DIR}/src/Utils.cu
    ${CMAKE_SOURCE_DIR}/src/Config/Config.cpp
    ${CMAKE_SOURCE_DIR}/src/exception/RalException.cpp
    ${CMAKE_SOURCE_DIR}/src/Traits/RuntimeTraits.cpp
    ${CMAKE_SOURCE_DIR}/src/operators/OrderBy.cpp
    ${cudf_adapter_source_files}
    ${binary_operation_source_files}
    ${communication_source_files}
    ${distribution_source_files}
) 

set(interpreter_tests_src
    ${source_files_src}
    interpreter_tests.cu
) 

configure_test(interpreter_tests "${interpreter_tests_src}")

 
set(project_tests_src
    ${source_files_src}
    process_project_tests.cu
) 

configure_test(project_tests "${project_tests_src}")


set(project_coalesce_tests_src
    ${source_files_src}
    interops_coalesce_test.cu
) 

configure_test(project_coalesce_tests "${project_coalesce_tests_src}")
