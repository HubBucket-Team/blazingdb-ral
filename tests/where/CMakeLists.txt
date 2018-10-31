set(where-test_SRCS
  ${CMAKE_SOURCE_DIR}/src/Utils.cu
  ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
  ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
  ${CMAKE_SOURCE_DIR}/src/ColumnManipulation.cu
  ${CMAKE_SOURCE_DIR}/src/StringUtil.cpp
  ${CMAKE_SOURCE_DIR}/src/JoinProcessor.cpp
  ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
  ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
  ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
  ${CMAKE_SOURCE_DIR}/src/CalciteInterpreter.cpp
  where-test.cpp
)

configure_test(where-test "${where-test_SRCS}")

target_link_libraries(where-test ral-test-utils)
