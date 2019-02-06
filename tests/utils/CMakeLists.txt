include_directories(
    ${CMAKE_SOURCE_DIR}/thirdparty/jitify
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper)
 

set(source_files_src
    ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
    ${CMAKE_SOURCE_DIR}/src/FreeMemory.cpp
    ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
    ${CMAKE_SOURCE_DIR}/src/cuDF/Allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/Utils.cu 
) 
 
set(utils_tests_src
    ${source_files_src}
    utils_tests.cu
) 

configure_test(utils_tests "${utils_tests_src}")

