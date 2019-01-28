set(jsource_files_SRC
  ${CMAKE_SOURCE_DIR}/src/CalciteInterpreter.cpp
  ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
  ${CMAKE_SOURCE_DIR}/src/ColumnManipulation.cu
  ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
  ${CMAKE_SOURCE_DIR}/src/FreeMemory.cpp
  ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
  ${CMAKE_SOURCE_DIR}/src/JoinProcessor.cpp
  ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
  ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
  ${CMAKE_SOURCE_DIR}/src/CodeTimer.cpp
  ${CMAKE_SOURCE_DIR}/src/cuDF/Allocator.cpp
  ${CMAKE_SOURCE_DIR}/src/Utils.cu
  ${binary_operation_source_files}
  pjoins.cu
)

configure_test(joint-test "${jsource_files_SRC}")
