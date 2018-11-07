cuda_add_library(tpch_interface
  ${CMAKE_SOURCE_DIR}/src/CalciteInterpreter.cpp
  ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
  ${CMAKE_SOURCE_DIR}/src/ColumnManipulation.cu
  ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
  ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
  ${CMAKE_SOURCE_DIR}/src/JoinProcessor.cpp
  ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
  ${CMAKE_SOURCE_DIR}/src/StringUtil.cpp
  ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
)
target_include_directories(tpch_interface PUBLIC .)
target_link_libraries(tpch_interface
  ${LIBGDF_STATIC_LIB}
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_NVRTC_LIBRARY}
  ${CUDA_NVTX_LIBRARY}
)

configure_test(tpch-test
  tpch-tests.cu
)
target_link_libraries(tpch-test tpch_interface)
