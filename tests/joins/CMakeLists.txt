cuda_add_library(join_interface
  ${CMAKE_SOURCE_DIR}/src/CalciteInterpreter.cpp
  ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp
  ${CMAKE_SOURCE_DIR}/src/ColumnManipulation.cu
  ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
  ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
  ${CMAKE_SOURCE_DIR}/src/JoinProcessor.cpp
  ${CMAKE_SOURCE_DIR}/src/LogicalFilter.cpp
  ${CMAKE_SOURCE_DIR}/src/StringUtil.cpp
  ${CMAKE_SOURCE_DIR}/src/CodeTimer.cpp
  ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
)
target_include_directories(join_interface PUBLIC .)
target_link_libraries(join_interface
  ${LIBGDF_STATIC_LIB}
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_NVRTC_LIBRARY}
  ${CUDA_NVTX_LIBRARY}
)

configure_test(fullouter_join-test
  fullouter_join.cu
)
target_link_libraries(fullouter_join-test join_interface)

configure_test(leftouter_join-test
  leftouter_join.cu
)
target_link_libraries(leftouter_join-test join_interface)
