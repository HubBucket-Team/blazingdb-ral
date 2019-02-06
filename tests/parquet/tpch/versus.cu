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

#include <rmm.h>


 #ifndef PARQUET_FILE_PATH
  #error PARQUET_FILE_PATH must be defined for precompiling
 #define PARQUET_FILE_PATH "/"
 #endif

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
            : filename(PARQUET_FILE_PATH) {}
 

    static constexpr std::size_t kGroups       = 1;
    static constexpr std::size_t kRowsPerGroup = 524289;
  

    //! Allocate a array of gdf columns to `gdf_columns` of `file_reade` filtering
    //  by row group indices and column indices

    template <ReaderType T, class ColumnReaderType, parquet::Type::type C>
    static inline gdf_error
    convert(gdf_column *column, ColumnReaderType *column_reader, int64_t amount_to_read){
        typedef typename parquet_traits<C>::parquet_type    parquet_type;
        parquet_type* values_buffer;
        gdf_valid_type* valid_bits;
        int16_t * definition_level;

        auto values_malloc_size = amount_to_read * sizeof(parquet_type);
        gdf_error status = Readers<T>::init_gdf_buffers((void **)&(values_buffer), &valid_bits, &definition_level, values_malloc_size, amount_to_read);

        std::int64_t levels_read;
        std::int64_t values_read = 0;
        std::int64_t nulls_count;
        auto batch_size = amount_to_read * 0.3;
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
        std::cout << "total_read|columntype: " << rows_read_total << "|" << typeid(ColumnReaderType).name() << std::endl;

        Readers<T>::buffer_to_gdf_column(column, (void *)values_buffer, valid_bits, values_malloc_size, amount_to_read, parquet_traits<C>::gdf_type());
        return GDF_SUCCESS;
    }

    template <ReaderType T>
    static inline gdf_error containerFrom(gdf_column *column, std::shared_ptr<parquet::ColumnReader> column_reader, int64_t numRecords) {

        parquet::Type::type parquetDataType = column_reader->type();

#define WHEN(dataType, Prefix)                                  \
        if ((dataType) == parquetDataType)                          \
            return convert<T, typename Readers<T>::Prefix##Reader, dataType>       \
                    (column, static_cast<typename Readers<T>::Prefix##Reader*>(column_reader.get()), numRecords)

        WHEN(parquet::Type::BOOLEAN, Bool);
        WHEN(parquet::Type::INT32, Int32);
        WHEN(parquet::Type::INT64, Int64);
        WHEN(parquet::Type::FLOAT, Float);
        WHEN(parquet::Type::DOUBLE, Double);

#undef WHEN

        std::cout << "ERROR: Bad parquet column type\n";
    }

    template <class T>
    static inline std::vector<const ::parquet::ColumnDescriptor *>
    _ColumnDescriptorsFrom(const T &file_reader, const std::vector<std::size_t> & indices) {
        const auto &row_group_reader = file_reader->RowGroup(0);

        std::vector<const parquet::ColumnDescriptor *> column_descriptors;
        column_descriptors.reserve(indices.size());

        for (const std::size_t i : indices) {
            column_descriptors.emplace_back(row_group_reader->Column(i)->descr());
        }

        return column_descriptors;
    }


    //! \returns a vector with the column indices of `raw_names` in `file_reader`
    template <class T>
    static inline std::vector<std::size_t>
    _GetColumnIndices(const T &file_reader, const char *const *const  raw_names) {

        std::vector<std::size_t> indices;

        const std::shared_ptr<const ::parquet::FileMetaData> &metadata =
        file_reader->metadata();

        const std::size_t num_columns =
        static_cast<std::size_t>(metadata->num_columns());

        auto schema = file_reader->RowGroup(0)->metadata()->schema();

        std::vector<std::pair<std::string, std::size_t>> parquet_columns;
        parquet_columns.reserve(num_columns);

        for (std::size_t i = 0; i < num_columns; i++) {
            if (schema->Column(i)->physical_type() != ::parquet::Type::BYTE_ARRAY
                && schema->Column(i)->physical_type()
                    != ::parquet::Type::FIXED_LEN_BYTE_ARRAY) {

                parquet_columns.push_back(
                std::make_pair(schema->Column(i)->name(), i));
            }
        }

        if (raw_names != nullptr) {
            for (const char *const *name_ptr = raw_names; *name_ptr != nullptr;
                name_ptr++) {

                std::string filter_name = *name_ptr;
                for (std::size_t i = 0; i < parquet_columns.size(); i++) {
                    if (filter_name == parquet_columns[i].first) {
                        indices.push_back(parquet_columns[i].second);
                        break;
                    }
                }
            }
        } else {
            for (std::size_t i = 0; i < parquet_columns.size(); i++) {
                indices.push_back(parquet_columns[i].second);
            }
        }
        return indices;
    }

    template <ReaderType T>
    inline static void
    readRowGroup(const std::unique_ptr<typename Readers<T>::FileReader> &file_reader) {

        const std::vector<std::size_t> column_indices = _GetColumnIndices(file_reader, nullptr);
        const std::vector<const parquet::ColumnDescriptor *> column_descriptors = _ColumnDescriptorsFrom(file_reader, column_indices);
        
        std::shared_ptr<parquet::FileMetaData> file_metadata = file_reader->metadata();
        const parquet::SchemaDescriptor *schema = file_metadata->schema();
        int numRowGroups = file_metadata->num_row_groups();

        std::vector<std::size_t> row_group_indices;
        for (std::size_t index = 0; index < numRowGroups; index++){
            row_group_indices.push_back(index);
        }

        int64_t num_rows = 0;
        for (std::size_t row_group_index_in_set = 0;
            row_group_index_in_set < row_group_indices.size();
            row_group_index_in_set++) {

            std::size_t row_group_index = row_group_indices[row_group_index_in_set];

            const auto row_group_reader = file_reader->RowGroup(static_cast<int>(row_group_index));

            num_rows += row_group_reader->metadata()->num_rows();
        }
        std::cout << "num_rows: " << num_rows << std::endl;
        const std::size_t num_columns = column_indices.size();
        std::cout << "num_columns: " << num_columns << std::endl;

        std::vector<gdf_column> columns;

        cudaStream_t cudaStream;
        cudaError_t  cudaError = cudaStreamCreate(&cudaStream);

        // if (_AllocateGdfColumns(file_reader,
        //                         row_group_indices,
        //                         column_indices,
        //                         cudaStream,
        //                         gdf_columns)
        //     != GDF_SUCCESS) {
        //     return GDF_FILE_ERROR;
        // }

        for (int rowGroupIndex = 0; rowGroupIndex < numRowGroups; rowGroupIndex++) {
            auto groupReader = file_reader->RowGroup(rowGroupIndex);
            for (int column_reader_index = 0; column_reader_index < num_columns; column_reader_index++) {
                auto columnIndex = column_indices[column_reader_index];
                const parquet::ColumnDescriptor *column = schema->Column(columnIndex);
                parquet::Type::type type = column->physical_type();

                const std::shared_ptr<parquet::ColumnReader> columnReader = groupReader->Column(columnIndex);

                gdf_column output;
                containerFrom<T>(&output, columnReader, num_rows); // @todo
                columns.push_back(output);
            }
        }
        cudaStreamDestroy(cudaStream);

    }
    const std::string filename;

    std::size_t columns_length = 0;
};


TEST_F(ParquetReaderAPITest, ParquetCpp) {
    std::unique_ptr<typename Readers<kParquet>::FileReader> reader = Readers<kParquet>::FileReader::OpenFile(filename);
    
    readRowGroup<kParquet>(reader);
    
}
// /home/aocsa/blazingsql/workspace/blazingdb-ral_project/blazingdb-io-parquet/blazingdb-ral/tests/io-test/lineitem_0_0.parquet
TEST_F(ParquetReaderAPITest, CuIOParquet) {
    std::unique_ptr<typename Readers<kGdf>::FileReader> reader = Readers<kGdf>::FileReader::OpenFile(filename);
    
    readRowGroup<kGdf>(reader);
    
}

 