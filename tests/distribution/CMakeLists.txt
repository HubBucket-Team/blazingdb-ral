set(distribution_primitives_test_SRCS
    distribution_primitives_test.cpp
    mocking/Client.cpp
    mocking/Server.cpp
)

configure_test(distribution_primitives_test "${distribution_primitives_test_SRCS}")
