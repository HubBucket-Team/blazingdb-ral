include_directories(
    ${CMAKE_SOURCE_DIR}/thirdparty/jitify
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper)

set(source_files
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/code/gdf-data.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/code/kernel.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/code/operation.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/code/traits.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/core/binop.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/core/launcher.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/util/operator.cpp
    ${CMAKE_SOURCE_DIR}/src/gdf_wrapper/binary-operation/jit/util/type.cpp
)

include_directories(${BLAZINGDB_IO_INCLUDEDIR} ${BLAZINGDB_IO_INCLUDE_DIR})

set(parse_csvtest_SRCS
  ${CMAKE_SOURCE_DIR}/src/GDFCounter.cu
  ${CMAKE_SOURCE_DIR}/src/CalciteExpressionParsing.cpp

  ${CMAKE_SOURCE_DIR}/src/cuDF/Allocator.cpp
  ${CMAKE_SOURCE_DIR}/src/GDFColumn.cu
  ${CMAKE_SOURCE_DIR}/src/io/data_parser/CSVParser.cpp
  ${CMAKE_SOURCE_DIR}/src/io/data_provider/UriDataProvider.cpp
  ${source_files}
)
   

set(HadoopFileSystemTest_Common_SRCS
    Common/SystemEnvironment.cpp
    Common/NativeHdfs.cpp
    Common/TestResources.cpp
    Common/HadoopFileSystemTest.cpp
)

set(HadoopFileSystemTest_SRCS
    ${HadoopFileSystemTest_Common_SRCS}
    ${parse_csvtest_SRCS}

    ConnectTest.cpp
    # DisconnectTest.cpp
    # ExistsTest.cpp
    # GetFileStatusTest.cpp
    # ListTest.cpp
    # MakeDirectoryTest.cpp
    # MoveTest.cpp
    # RemoveTest.cpp
    # OpenReadableTest.cpp
    # OpenWriteableTest.cpp
    # TruncateTest.cpp
)

configure_test(hdfs-test "${HadoopFileSystemTest_SRCS}")
