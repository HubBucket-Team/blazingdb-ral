set(resultset_SRCS
    ${CMAKE_SOURCE_DIR}/src/Utils.cu
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
    ${CMAKE_SOURCE_DIR}/src/ColumnManipulation.cu
    ${CMAKE_SOURCE_DIR}/src/CalciteInterpreter.cpp
    ${CMAKE_SOURCE_DIR}/src/StringUtil.cpp
    ${CMAKE_SOURCE_DIR}/src/JoinProcessor.cpp
    ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
    ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
    resultset.cpp)
configure_test(resultset "${resultset_SRCS}")

target_link_libraries(resultset ral-test-utils)