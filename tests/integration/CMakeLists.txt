set(integration_test_SRCS
    integration-test.cu
)

configure_test(integration-test "${integration_test_SRCS}")

add_subdirectory(aggregation-without-group-by)
