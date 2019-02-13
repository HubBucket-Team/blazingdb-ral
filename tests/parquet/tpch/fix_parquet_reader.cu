/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
 *     Copyright 2018 William Malpica <william@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mutex>
#include <numeric>
#include <thread>


#include <cassert>

#include <GDFColumn.cuh>

#include <arrow/util/bit-util.h>
#include <arrow/util/logging.h>

#include <parquet/column_reader.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>

#include <thrust/device_ptr.h>

#include "cuio/parquet/column_reader.h"
#include "cuio/parquet/file_reader.h"
#include "cuio/parquet/util/bit_util.cuh"

#include <rmm.h>

#include <cuio/parquet/api.h>

#include <arrow/io/file.h>
#include <arrow/util/logging.h>

#include <boost/filesystem.hpp>

#include <parquet/column_writer.h>
#include <parquet/file_writer.h>
#include <parquet/properties.h>
#include <parquet/schema.h>
#include <parquet/types.h>

#include <gtest/gtest.h>


 #ifndef PARQUET_FILE_PATH
  #error PARQUET_FILE_PATH must be defined for precompiling
 #define PARQUET_FILE_PATH "/"
 #endif

// #include "utils.cuh"

// NOTE c.gonzales percy para el parquet file reader aqui los detalles del file 
//file:reader-test.parquet
//Total RowCount: 50000

/* NOTE Meta info for file:reader-test.parquet
file:          file:reader-test.parquet 
creator:       parquet-cpp version 1.4.0 

file schema:   schema 
--------------------------------------------------------------------------------
boolean_field: REQUIRED BOOLEAN R:0 D:0
int64_field:   REQUIRED INT64 R:0 D:0
double_field:  REQUIRED DOUBLE R:0 D:0

row group 1:   RC:50000 TS:767525 OFFSET:4 
--------------------------------------------------------------------------------
boolean_field:  BOOLEAN SNAPPY DO:0 FPO:4 SZ:329/6281/19.09 VC:50000 ENC:PLAIN,RLE
int64_field:    INT64 SNAPPY DO:393 FPO:354318 SZ:454082/500168/1.10 VC:50000 ENC:PLAIN,RLE,PLAIN_DICTIONARY
double_field:   DOUBLE SNAPPY DO:454570 FPO:667527 SZ:313114/500168/1.60 VC:50000 ENC:PLAIN,RLE,PLAIN_DICTIONARY
*/

/* NOTE Schema for file:reader-test.parquet
message schema {
  required boolean boolean_field;
  required int64 int64_field;
  required double double_field;
}

*/


class ParquetReaderAPITest : public testing::Test {
protected:
    ParquetReaderAPITest()
      : filename(PARQUET_FILE_PATH) {}

  
    const std::string filename;

    gdf_column *columns        = nullptr;
    std::size_t columns_length = 0;
};



TEST_F(ParquetReaderAPITest, ReadAll) {
    std::cout << "reading..." << PARQUET_FILE_PATH << std::endl;
    gdf_error error_code = gdf::parquet::read_parquet(filename.c_str(), nullptr, &columns, &columns_length);
    for (size_t i=0;i < columns_length; i++) {
      print_gdf_column(&columns[i]);
    }

    EXPECT_EQ(GDF_SUCCESS, error_code);
 
}