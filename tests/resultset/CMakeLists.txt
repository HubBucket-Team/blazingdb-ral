set(resultset_SRCS
    ${CMAKE_SOURCE_DIR}/src/Utils.cu
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${CMAKE_SOURCE_DIR}/src/ResultSetRepository.cpp
    resultset.cpp)
configure_test(resultset "${resultset_SRCS}")