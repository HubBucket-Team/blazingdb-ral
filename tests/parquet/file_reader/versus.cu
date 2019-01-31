/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa alexander@blazingdb.com
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

#include <cudf.h>

#include <arrow/io/file.h>
#include <arrow/util/logging.h>


#include <parquet/column_writer.h>
#include <parquet/file_writer.h>
#include <parquet/properties.h>
#include <parquet/schema.h>
#include <parquet/types.h>

#include <gtest/gtest.h>
#include <parquet/api/writer.h>
#include <parquet/api/reader.h>

#include "cudf/types.h"

#include "cuio/parquet/util/bit_util.cuh"
#include "cuio/parquet/column_reader.h"
#include "cuio/parquet/file_reader.h"

enum ReaderType : std::uint8_t { kGdf, kParquet };

template <ReaderType T>
struct Readers {};

template <>
struct Readers<kParquet> {
    typedef typename parquet::BoolReader          BoolReader;
    typedef typename parquet::Int32Reader         Int32Reader;
    typedef typename parquet::Int64Reader         Int64Reader;
    typedef typename parquet::FloatReader         FloatReader;
    typedef typename parquet::DoubleReader        DoubleReader;
    typedef typename parquet::ParquetFileReader   FileReader;

    static inline gdf_error init_gdf_buffers(void **host_values, gdf_valid_type** host_valid, int16_t** def_levels, uint32_t values_malloc_size, gdf_size_type column_size){
       *host_values = malloc(values_malloc_size);
       auto n_bytes = gdf::util::PaddedLength(arrow::BitUtil::BytesForBits(column_size));
       *host_valid = (gdf_valid_type*)malloc(n_bytes);
       *def_levels = (int16_t*)malloc(column_size * sizeof(int16_t));
        return GDF_SUCCESS;
    }

    static inline gdf_error buffer_to_gdf_column(gdf_column *output, void *host_values, gdf_valid_type* host_valid, uint32_t values_malloc_size, gdf_size_type column_size, gdf_dtype dtype) {
       void *device_values;
       cudaError_t cuda_error = cudaMalloc((void **)&device_values, values_malloc_size);
       cudaMemcpy(device_values, host_values, values_malloc_size, cudaMemcpyHostToDevice);

       gdf_valid_type *device_valid;
       auto n_bytes =  gdf::util::PaddedLength(arrow::BitUtil::BytesForBits(column_size));
       cudaMalloc((void **)&device_valid, n_bytes);
       cudaMemcpy(device_valid, host_valid, n_bytes, cudaMemcpyHostToDevice);

       //auto zero_bits = count_zero_bits(host_valid, column_size);

       free(host_values);
       free(host_valid);
       return gdf_column_view(output, device_values, device_valid, column_size, dtype);
    }
};

 
template <>                                                            
struct Readers<kGdf> {
    typedef typename gdf::parquet::BoolReader   BoolReader;            
    typedef typename gdf::parquet::Int32Reader  Int32Reader;           
    typedef typename gdf::parquet::Int64Reader  Int64Reader;           
    typedef typename gdf::parquet::FloatReader  FloatReader;          
    typedef typename gdf::parquet::DoubleReader DoubleReader;          
    typedef typename gdf::parquet::FileReader   FileReader;            
     

    static inline gdf_error init_gdf_buffers(void **device_values, gdf_valid_type** device_valid, int16_t** def_levels, uint32_t values_malloc_size, gdf_size_type column_size){
        cudaError_t cuda_error = cudaMalloc(device_values, values_malloc_size);
         auto n_bytes =  gdf::util::PaddedLength(arrow::BitUtil::BytesForBits(column_size));
        cudaMalloc(device_valid, n_bytes);
        cudaMalloc(def_levels, column_size * sizeof(int16_t));
        return GDF_SUCCESS;
    }
     

    static inline gdf_error buffer_to_gdf_column(gdf_column *output, void *device_values, gdf_valid_type* device_valid, uint32_t values_malloc_size, gdf_size_type column_size, gdf_dtype dtype) {
        return gdf_column_view(output, device_values, device_valid, column_size, dtype);
    }
};

template<parquet::Type::type T>
struct parquet_traits {

};

#define PARQUET_TRAITS_FACTORY(ParquetTypeEnum, ParquetType, GdfTypeValue)  \
    template <> struct parquet_traits<ParquetTypeEnum> {                        \
        typedef ParquetType             parquet_type;                           \
        static inline gdf_dtype         gdf_type() { return GdfTypeValue; }       \
    }

PARQUET_TRAITS_FACTORY(parquet::Type::BOOLEAN, bool, GDF_INT8);
PARQUET_TRAITS_FACTORY(parquet::Type::INT32, int32_t, GDF_INT32);
PARQUET_TRAITS_FACTORY(parquet::Type::INT64, int64_t, GDF_INT64);
PARQUET_TRAITS_FACTORY(parquet::Type::FLOAT, float, GDF_FLOAT32);
PARQUET_TRAITS_FACTORY(parquet::Type::DOUBLE, double, GDF_FLOAT64);

#undef PARQUET_TRAITS_FACTORY


class ParquetReaderAPITest : public testing::Test {
protected:
    ParquetReaderAPITest()
            : filename("/tmp/sample.parquet") {}

    std::int32_t
    genInt32(int i) {
        if (i >= 100 && i < 150) {
            return 10000;
        } else if (i >= 200 && i < 300) {
            return 20000;
        } else if (i >= 310 && i < 350) {
            return 30000;
        } else if (i >= 450 && i < 550) {
            return 40000;
        } else if (i >= 800 && i < 950) {
            return 50000;
        } else {
            return i * 100;
        }
    }

    std::int64_t
    genInt64(int i) {
        if (i >= 100 && i < 150) {
            return 10000;
        } else if (i >= 200 && i < 300) {
            return 20000;
        } else if (i >= 310 && i < 350) {
            return 30000;
        } else if (i >= 450 && i < 550) {
            return 40000;
        } else if (i >= 800 && i < 950) {
            return 50000;
        } else {
            return i * 100000;
        }
    }

    static constexpr std::size_t kGroups       = 1;
    static constexpr std::size_t kRowsPerGroup = 524289;

    void
    SetUp() final {
        try {

            std::shared_ptr<::arrow::io::FileOutputStream> stream;
            PARQUET_THROW_NOT_OK(
                    ::arrow::io::FileOutputStream::Open(filename, &stream));

            std::shared_ptr<::parquet::schema::GroupNode> schema =
                    CreateSchema();

            ::parquet::WriterProperties::Builder builder;
            builder.compression(::parquet::Compression::SNAPPY);
            std::shared_ptr<::parquet::WriterProperties> properties =
                    builder.build();

            std::shared_ptr<::parquet::ParquetFileWriter> file_writer =
                    ::parquet::ParquetFileWriter::Open(stream, schema, properties);

            std::int16_t repetition_level = 0;

            for (std::size_t i = 0; i < kGroups; i++) {
                ::parquet::RowGroupWriter *row_group_writer =
                        file_writer->AppendRowGroup(kRowsPerGroup);

                ::parquet::BoolWriter *bool_writer =
                        static_cast<::parquet::BoolWriter *>(
                                row_group_writer->NextColumn());
                for (std::size_t j = 0; j < kRowsPerGroup; j++) {
                    int ind = i * kRowsPerGroup + j;
                    std::int16_t definition_level = ind % 3 > 0 ? 1 : 0;
                    bool         bool_value       = true;
                    bool_writer->WriteBatch(
                            1, &definition_level, &repetition_level, &bool_value);
                }

                ::parquet::Int32Writer *int32_writer =
                        static_cast<::parquet::Int32Writer *>(
                                row_group_writer->NextColumn());
                for (std::size_t j = 0; j < kRowsPerGroup; j++) {
                    int ind = i * kRowsPerGroup + j;
                    std::int16_t definition_level = ind % 3 > 0 ? 1 : 0;
                    std::int32_t int32_value = genInt32(ind);
                    int32_writer->WriteBatch(
                            1, &definition_level, &repetition_level, &int32_value);
                }

                ::parquet::Int64Writer *int64_writer =
                        static_cast<::parquet::Int64Writer *>(
                                row_group_writer->NextColumn());
                for (std::size_t j = 0; j < kRowsPerGroup; j++) {
                    int ind = i * kRowsPerGroup + j;
                    std::int16_t definition_level = ind % 3 > 0 ? 1 : 0;
                    std::int64_t int64_value = genInt64(ind);
                    int64_writer->WriteBatch(
                            1, &definition_level, &repetition_level, &int64_value);
                }

                ::parquet::DoubleWriter *double_writer =
                        static_cast<::parquet::DoubleWriter *>(
                                row_group_writer->NextColumn());
                for (std::size_t j = 0; j < kRowsPerGroup; j++) {
                    int ind = i * kRowsPerGroup + j;
                    std::int16_t definition_level = ind % 3 > 0 ? 1 : 0;
                    double       double_value     = (double)ind;
                    double_writer->WriteBatch(
                            1, &definition_level, &repetition_level, &double_value);
                }
            }

            file_writer->Close();

            DCHECK(stream->Close().ok());
        } catch (const std::exception &e) {
            FAIL() << "Generate file" << e.what();
        }
    }

    std ::shared_ptr<::parquet::schema::GroupNode>
    CreateSchema() {
        return std::static_pointer_cast<::parquet::schema::GroupNode>(
                ::parquet::schema::GroupNode::Make(
                        "schema",
                        ::parquet::Repetition::REQUIRED,
                        ::parquet::schema::NodeVector{
                                ::parquet::schema::PrimitiveNode::Make(
                                        "boolean_field",
                                        ::parquet::Repetition::OPTIONAL,
                                        ::parquet::Type::BOOLEAN,
                                        ::parquet::LogicalType::NONE),
                                ::parquet::schema::PrimitiveNode::Make(
                                        "int32_field",
                                        ::parquet::Repetition::OPTIONAL,
                                        ::parquet::Type::INT32,
                                        ::parquet::LogicalType::NONE),
                                ::parquet::schema::PrimitiveNode::Make(
                                        "int64_field",
                                        ::parquet::Repetition::OPTIONAL,
                                        ::parquet::Type::INT64,
                                        ::parquet::LogicalType::NONE),
                                ::parquet::schema::PrimitiveNode::Make(
                                        "double_field",
                                        ::parquet::Repetition::OPTIONAL,
                                        ::parquet::Type::DOUBLE,
                                        ::parquet::LogicalType::NONE),
                        }));
    }

    void
    TearDown() final {
        //if (std::remove(filename.c_str())) { FAIL() << "Remove file"; }
    }






    template <ReaderType T, class ColumnReaderType, parquet::Type::type C>
    static inline gdf_error
    convert(gdf_column *column, ColumnReaderType *column_reader, int64_t amount_to_read, uint32_t batch_size) {
        typedef typename parquet_traits<C>::parquet_type    parquet_type;
        parquet_type* values_buffer;
        gdf_valid_type* valid_bits;
        int16_t * definition_level;

        auto values_malloc_size = amount_to_read * sizeof(parquet_type);
        gdf_error status = Readers<T>::init_gdf_buffers((void **)&(values_buffer), &valid_bits, &definition_level, values_malloc_size, amount_to_read);

        std::int64_t levels_read;
        std::int64_t values_read = 0;
        std::int64_t nulls_count;

        int64_t rows_read_total = 0;
        while (column_reader->HasNext() && rows_read_total < amount_to_read) {
            int64_t rows_read = column_reader->ReadBatchSpaced(batch_size,
                                                               &definition_level[rows_read_total],
                                                               nullptr,
                                                               &values_buffer[rows_read_total],
                                                               valid_bits,
                                                               0,
                                                               &levels_read,
                                                               &values_read,
                                                               &nulls_count);
            rows_read_total += rows_read;
        }
        std::cout << "columntype: " << typeid(ColumnReaderType).name() << std::endl;
        Readers<T>::buffer_to_gdf_column(column, (void *)values_buffer, valid_bits, values_malloc_size, amount_to_read, parquet_traits<C>::gdf_type());
        return GDF_SUCCESS;
    }

    template <ReaderType T>
    static inline gdf_error containerFrom(gdf_column *column, std::shared_ptr<parquet::ColumnReader> column_reader, int64_t numRecords, uint32_t batch_size) {

        parquet::Type::type parquetDataType = column_reader->type();

#define WHEN(dataType, Prefix)                                  \
        if ((dataType) == parquetDataType)                          \
            return convert<T, typename Readers<T>::Prefix##Reader, dataType>       \
                    (column, static_cast<typename Readers<T>::Prefix##Reader*>(column_reader.get()), numRecords, batch_size)

        WHEN(parquet::Type::BOOLEAN, Bool);
        WHEN(parquet::Type::INT32, Int32);
        WHEN(parquet::Type::INT64, Int64);
        WHEN(parquet::Type::FLOAT, Float);
        WHEN(parquet::Type::DOUBLE, Double);

#undef WHEN

        std::cout << "ERROR: Bad parquet column type\n";
    }

    template <ReaderType T>
    inline static void
    readRowGroup(const std::unique_ptr<typename Readers<T>::FileReader> &parquet_reader, uint32_t batch_size) {

        std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
        const parquet::SchemaDescriptor *schema = file_metadata->schema();
        int numRowGroups = file_metadata->num_row_groups();

        std::vector<gdf_column> columns;

        for (int rowGroupIndex = 0; rowGroupIndex < numRowGroups; rowGroupIndex++) {
            auto groupReader = parquet_reader->RowGroup(rowGroupIndex);
            const parquet::RowGroupMetaData *rowGroupMetadata = groupReader->metadata();
            for (int columnIndex = 0; columnIndex < file_metadata->num_columns(); columnIndex++) {
                const parquet::ColumnDescriptor *column = schema->Column(columnIndex);
                std::unique_ptr<parquet::ColumnChunkMetaData> columnMetaData = rowGroupMetadata->ColumnChunk(
                        columnIndex);
                parquet::Type::type type = column->physical_type();

                if (type != parquet::Type::BYTE_ARRAY){
                    const std::shared_ptr<parquet::ColumnReader> columnReader = groupReader->Column(columnIndex);
                    int64_t numRecords = rowGroupMetadata->num_rows();

                    gdf_column output;
                    containerFrom<T>(&output, columnReader, numRecords, batch_size);
                    columns.push_back(output);
                }
            }
        }
    }
    const std::string filename;

    std::size_t columns_length = 0;
};

TEST_F(ParquetReaderAPITest, ParquetCpp) {
    std::unique_ptr<typename Readers<kParquet>::FileReader> reader = Readers<kParquet>::FileReader::OpenFile(filename);
    
    readRowGroup<kParquet>(reader, kRowsPerGroup);
    
}
TEST_F(ParquetReaderAPITest, CuIOParquetCpp) {
    std::unique_ptr<typename Readers<kGdf>::FileReader> reader = Readers<kGdf>::FileReader::OpenFile(filename);
    
    readRowGroup<kGdf>(reader, kRowsPerGroup);
    
}

 