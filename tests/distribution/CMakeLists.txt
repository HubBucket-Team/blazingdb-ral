
set(test_name "distribution_primitives_test")

project(${test_name} CUDA CXX)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")

set(test_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/distribution_primitives_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mocking/Client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mocking/Server.cpp
    ${CMAKE_SOURCE_DIR}/src/communication/adapter/GpuFunctionsAdapter.cpp
    ${CMAKE_SOURCE_DIR}/src/communication/factory/MessageFactory.cpp
    ${CMAKE_SOURCE_DIR}/src/communication/CommunicationData.cpp
    ${CMAKE_SOURCE_DIR}/src/cuDF/generator/sample_generator.cu
    ${CMAKE_SOURCE_DIR}/src/cuDF/Allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/distribution/Exception.cpp
    ${CMAKE_SOURCE_DIR}/src/distribution/NodeColumns.cpp
    ${CMAKE_SOURCE_DIR}/src/distribution/NodeSamples.cpp
    ${CMAKE_SOURCE_DIR}/src/distribution/primitives.cpp
    ${CMAKE_SOURCE_DIR}/src/exception/RalException.cpp
    ${CMAKE_SOURCE_DIR}/src/Traits/RuntimeTraits.cpp
    ${CMAKE_SOURCE_DIR}/src/FreeMemory.cpp
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
)

add_executable(${test_name} ${test_sources})

target_include_directories(${test_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mock)
target_include_directories(${test_name} PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(
    ${test_name}
    GTest::GTest
    GTest::Main
    gmock
    gmock_main
    ${LIBGDF_STATIC_LIB}
    NVStrings
    rmm
    ${BLAZINGDB_COMMUNICATION_STATIC_LIB}
    Boost::filesystem
    Boost::system
)

set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")

add_test(NAME ${test_name} COMMAND ${test_name})
