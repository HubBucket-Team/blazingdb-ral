set(PARQUET_DIR ${CMAKE_SOURCE_DIR}/src/parquet)

set(source_files
    ${PARQUET_DIR}/file_reader.cpp
    ${PARQUET_DIR}/file_reader_contents.cpp
    ${PARQUET_DIR}/row_group_reader_contents.cpp
    ${PARQUET_DIR}/column_reader.cu
)
