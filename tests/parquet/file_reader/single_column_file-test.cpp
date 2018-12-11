/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
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

#include <sstream>

#include <boost/filesystem.hpp>

#include <arrow/io/file.h>
#include <arrow/util/logging.h>

#include <parquet/properties.h>
#include <parquet/schema.h>
#include <parquet/types.h>

#include <gtest/gtest.h>

#include "src/parquet/file_reader.h"
#include "src/parquet/column_reader.h"

#include <cudf.h>


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

class SingleColumnFileTest : public testing::Test {
protected:
    SingleColumnFileTest();

    virtual void SetUp() override;
    virtual void TearDown() override;

    static constexpr std::size_t kRowsPerGroup = 100;

    std::string filename;
};

using Types = ::testing::Types<::parquet::BooleanType,
                               ::parquet::Int32Type,
                               ::parquet::Int64Type,
                               ::parquet::FloatType,
                               ::parquet::DoubleType>;

TYPED_TEST_CASE(SingleColumnFileTest, Types);

SingleColumnFileTest::SingleColumnFileTest() {
}

void SingleColumnFileTest::SetUp() {
    filename = std::string(PARQUET_FILE_PATH);
    
    std::cout << "READING PARQUET FILE: " << filename << std::endl;
}

void SingleColumnFileTest::TearDown() {
}

TEST_F(SingleColumnFileTest, ReadAll) {
    std::unique_ptr<gdf::parquet::FileReader> reader =
      gdf::parquet::FileReader::OpenFile(this->filename);

    std::shared_ptr<gdf::parquet::ColumnReader<parquet::DoubleType>> column_reader =
      std::static_pointer_cast<gdf::parquet::ColumnReader<parquet::DoubleType>>(
        reader->RowGroup(0)->Column(0));

    ASSERT_TRUE(column_reader->HasNext());

    const std::size_t rowsPerGroup = this->kRowsPerGroup;

    gdf_column column{
      .data       = new std::uint8_t[rowsPerGroup * sizeof(double)],
      .valid      = new std::uint8_t[rowsPerGroup],
      .size       = 0,
      .dtype      = GDF_invalid,
      .null_count = 0,
      .dtype_info = {},
    };

    std::int16_t definition_levels[rowsPerGroup];
    std::int16_t repetition_levels[rowsPerGroup];

    const std::size_t total_read = column_reader->ToGdfColumn(column);

    EXPECT_EQ(rowsPerGroup, total_read);

    double val = 0.001;
    const double h = 0.001;

    for (std::size_t i = 0; i < rowsPerGroup; i++) {
        double expected = val;
        std::int64_t value = static_cast<double *>(column.data)[i];
       
        EXPECT_EQ(expected, value);
        
        val += h;
    }

    delete[] static_cast<std::uint8_t *>(column.data);
    delete[] column.valid;
}
