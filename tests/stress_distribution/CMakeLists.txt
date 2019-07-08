set(stress_distribution_test_SRCS
    send_samples_test.cu
    ${CMAKE_SOURCE_DIR}/tests/utilities/gdf_column_cpp_utilities.cu
    ${CMAKE_SOURCE_DIR}/src/Traits/RuntimeTraits.cpp
)

configure_test(stress_distribution_test "${stress_distribution_test_SRCS}")
