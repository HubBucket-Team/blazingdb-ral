set(inner-join_SRCS
    ${CMAKE_SOURCE_DIR}/src/Utils.cu
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${CMAKE_SOURCE_DIR}/src/ColumnManipulation.cu
    ${CMAKE_SOURCE_DIR}/src/StringUtil.cpp
    ${CMAKE_SOURCE_DIR}/src/CodeTimer.cpp
    ${CMAKE_SOURCE_DIR}/src/JoinProcessor.cpp
    ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
    ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
    ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
    ${CMAKE_SOURCE_DIR}/src/CalciteInterpreter.cpp
    inner-join.cu)
configure_test(inner-join "${inner-join_SRCS}")


set(left-join_SRCS
    ${CMAKE_SOURCE_DIR}/src/Utils.cu
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${CMAKE_SOURCE_DIR}/src/ColumnManipulation.cu
    ${CMAKE_SOURCE_DIR}/src/StringUtil.cpp
    ${CMAKE_SOURCE_DIR}/src/CodeTimer.cpp
    ${CMAKE_SOURCE_DIR}/src/JoinProcessor.cpp
    ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
    ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
    ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
    ${CMAKE_SOURCE_DIR}/src/CalciteInterpreter.cpp
    left-join.cu)
configure_test(left-join "${left-join_SRCS}")
