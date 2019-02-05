#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
 
set(parquet_versus-test_SRCS
    ${parquet_reader_source_files}
    versus.cu
)

set (parquet_bug_fix-test_SRCS
    ${parquet_reader_source_files}
    fix_parquet_reader.cu
)


configure_test(parquet_bug_fix-test "${parquet_bug_fix-test_SRCS}")
configure_test(parquet_versus-test "${parquet_versus-test_SRCS}")

 
set(parquet_file_PATH
    ${CMAKE_SOURCE_DIR}/tests/parquet/tpch/data/nation_0_0.parquet)

target_compile_definitions(parquet_bug_fix-test
    PUBLIC -DPARQUET_FILE_PATH="${parquet_file_PATH}")
 

target_compile_definitions(parquet_versus-test
    PUBLIC -DPARQUET_FILE_PATH="${parquet_file_PATH}")
