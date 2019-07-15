
set(test_name "sample_generator_test")

project(${test_name} CXX CUDA)

set(test_sources ${CMAKE_CURRENT_SOURCE_DIR}/random_generator_test.cu)

add_executable(${test_name} ${test_sources})

target_include_directories(${test_name} PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(${test_name} ${CUDA_CUDA_LIBRARY} GTest::GTest GTest::Main)

set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")

add_test(NAME ${test_name} COMMAND ${test_name})
