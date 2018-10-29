cuda_add_library(ral-test-utils
  ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
  ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
)
target_include_directories(ral-test-utils PUBLIC .)
target_link_libraries(ral-test-utils
  ${LIBGDF_STATIC_LIB}
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_NVRTC_LIBRARY}
  ${CUDA_NVTX_LIBRARY}
)

configure_test(utils-test
  utils-test.cpp
)
target_link_libraries(utils-test ral-test-utils)
