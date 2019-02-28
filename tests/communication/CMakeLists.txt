set(test_name "component_message_test")

project(${test_name} CXX CUDA)

set(test_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/component_message_test.cpp
    ${CMAKE_SOURCE_DIR}/src/cuDF/Allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/communication/adapter/GpuFunctionsAdapter.cpp
    ${CMAKE_SOURCE_DIR}/src/communication/factory/MessageFactory.cpp
    ${CMAKE_SOURCE_DIR}/src/communication/network/Client.cpp
    ${CMAKE_SOURCE_DIR}/src/communication/network/Server.cpp
    ${CMAKE_SOURCE_DIR}/src/Traits/RuntimeTraits.cpp
    ${CMAKE_SOURCE_DIR}/src/FreeMemory.cpp
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
)

add_executable(${test_name} ${test_sources})

target_include_directories(${test_name} PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(${test_name}
                      ${CUDA_CUDA_LIBRARY}
                      GTest::GTest
                      GTest::Main
                      rmm
                      ${LIBGDF_STATIC_LIB}
                      ${BLAZINGDB_COMMUNICATION_STATIC_LIB}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      gcov
)

set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")

add_test(NAME ${test_name} COMMAND ${test_name})
