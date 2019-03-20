set(distribution_primitives_test_SRCS
    distribution_primitives_test.cpp
    mocking/Client.cpp
    mocking/Server.cpp
    ${CMAKE_SOURCE_DIR}/tests/utilities/gdf_column_cpp_utilities.cu
    ${CMAKE_SOURCE_DIR}/src/Traits/RuntimeTraits.cpp
)

configure_test(distribution_primitives_test "${distribution_primitives_test_SRCS}")
