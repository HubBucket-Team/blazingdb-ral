add_custom_target(gdf-ref-counter-tests)

set(SRCS ${cudf_adapter_source_files} ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    gdf-ref-counter-contains-one-test.cu
    gdf-ref-counter-contains-test.cu
    gdf-ref-counter-counting-test.cu
    gdf-ref-counter-no-side-effects-test.cu
    gdf-ref-counter-test.h
    gdf-ref-counter-threading-test.cu)

# TODO fix these tests
#configure_test(gdf-ref-counter-test "${SRCS}")
