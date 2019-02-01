set(PARQUET_DIR ${CMAKE_SOURCE_DIR}/src/cuio/parquet)

set(source_files
    ${PARQUET_DIR}/file_reader.cpp
    ${PARQUET_DIR}/file_reader_contents.cpp
    ${PARQUET_DIR}/row_group_reader_contents.cpp
    ${PARQUET_DIR}/column_reader.cu
    ${PARQUET_DIR}/decoder/cu_level_decoder.cu
    ${PARQUET_DIR}/page_reader.cpp
    ${PARQUET_DIR}/decoder/arrow/cu_decoder.cu
    ${PARQUET_DIR}/decoder/arrow/util/pinned_allocator.cu
    ${PARQUET_DIR}/api.cpp
)
