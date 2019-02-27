set(calcite_interpreter_test_SRCS
    calcite_interpreter-test.cu
)

set(calcite_interpreter_where_test_SRCS
    calcite_interpreter-where-test.cu
)

configure_test(calcite-interpreter-test "${calcite_interpreter_test_SRCS}")
configure_test(calcite-interpreter-where-test "${calcite_interpreter_where_test_SRCS}")


target_link_libraries(calcite-interpreter-test
                      ${cuDF_adapter_testing_library}
                      ${cuDF_adapter_testing_dependencies}
)
