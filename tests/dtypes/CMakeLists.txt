set(dtypes-test_SRCS
  ${CMAKE_SOURCE_DIR}/src/CalciteInterpreter.cpp
  ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
  ${CMAKE_SOURCE_DIR}/src/ColumnManipulation.cu
  ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
  ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
  ${CMAKE_SOURCE_DIR}/src/JoinProcessor.cpp
  ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
  ${CMAKE_SOURCE_DIR}/src/StringUtil.cpp
  ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
  dtypes-test.cu)
configure_test(dtypes-test "${dtypes-test_SRCS}")
