set(uu_test_SRCS
    ${CMAKE_SOURCE_DIR}/src/cuDF/Allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu

    utils-test.cpp
)

#configure_test(utils-test ${uu_test_SRCS})
