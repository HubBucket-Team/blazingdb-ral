set(source_files
    tpch-tests.cu
)

set(tpch_csv_tests_src
    tpch_csv_tests.cu
)

configure_test(tpch-test "${source_files}")
configure_test(tpch_csv_tests "${tpch_csv_tests_src}")
