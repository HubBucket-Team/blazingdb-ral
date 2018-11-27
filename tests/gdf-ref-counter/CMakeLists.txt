add_custom_target(gdf-ref-counter-tests)

function(add_gdf_ref_counter_test NAME)
set(NAME "gdf-ref-counter-${NAME}-test")
set(SRCS ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu "${NAME}.cu")
add_dependencies(gdf-ref-counter-tests ${NAME})
configure_test(${NAME} "${SRCS}")
target_link_libraries(${NAME}
                      ${cuDF_adapter_testing_library}
                      ${cuDF_adapter_testing_dependencies}
)
endfunction()

add_gdf_ref_counter_test(contains)
add_gdf_ref_counter_test(contains-one)
add_gdf_ref_counter_test(counting)
add_gdf_ref_counter_test(no-side-effects)
add_gdf_ref_counter_test(threading)
