set(test_SRCS
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${cudf_adapter_source_files}
    utils-test.cpp
)

configure_test(utils-test ${test_SRCS})
