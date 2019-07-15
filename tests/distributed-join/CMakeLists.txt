set(distributed_join_test_sources
    distributed_join_test.cpp
    ${CMAKE_SOURCE_DIR}/tests/utilities/gdf_column_cpp_utilities.cu
)

configure_test(distributed_join_test "${distributed_join_test_sources}")
