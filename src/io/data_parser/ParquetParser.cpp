/*
 * ParquetParser.cpp
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#include "ParquetParser.h"


#include <cuio/parquet/api.h>

#include <arrow/io/file.h>
#include <arrow/util/logging.h>

#include <boost/filesystem.hpp>

#include <parquet/column_writer.h>
#include <parquet/file_writer.h>
#include <parquet/properties.h>
#include <parquet/schema.h>
#include <parquet/types.h>

#include <GDFColumn.cuh>
#include <GDFCounter.cuh>

#include <cudf.h>
#include "../Utils.cuh"
#include "ParserUtils.h"

namespace ral {
namespace io {

parquet_parser::parquet_parser() {
	// TODO Auto-generated constructor stub

}

parquet_parser::~parquet_parser() {
	// TODO Auto-generated destructor stub
}

void parquet_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file, std::vector<gdf_column_cpp> & columns) {
	// size_t num_row_groups;
	// size_t num_cols;
	// std::vector< gdf_dtype> dtypes;
	// std::vector< std::string> column_names;

 	std::vector<bool> 	include_columns;
	// gdf_error error = gdf::parquet::read_schema(file, num_row_groups, num_cols, dtypes, column_names, include_columns);
	this->parse(file, columns, include_columns);
}

void parquet_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			std::vector<gdf_column_cpp> & gdf_columns_out,
			std::vector<bool> include_columns){
	// NOTE: Not sure but seems HOST_BUFFER is not supported
	int64_t num_bytes;
	file->GetSize(&num_bytes);

	std::vector<uint8_t> byteData(num_bytes);
	CUDF_CALL( read_file_into_buffer(file, num_bytes, byteData.data(), 100, 10) );

	pq_read_arg readerArgs;
	readerArgs.source_type = HOST_BUFFER;
  readerArgs.source = reinterpret_cast<char*>(byteData.data());
  readerArgs.buffer_size = byteData.size();
  readerArgs.use_cols = nullptr;
  // readerArgs.use_cols_len;

	CUDF_CALL( read_parquet(&readerArgs) );

	gdf_columns_out.resize(readerArgs.num_cols_out);
 	for(size_t i = 0; i < gdf_columns_out.size(); i++){
 		gdf_columns_out[i].create_gdf_column(readerArgs.data[i]);
	}

	free(readerArgs.data);
	free(readerArgs.index_col);
}

void parquet_parser::parse_schema(std::shared_ptr<arrow::io::RandomAccessFile> file, std::vector<gdf_column_cpp> & gdf_columns_out)  {
	// NOTE: we are reading the whole file into memory, we should read the only needed data for the metada as in
	// FileReaderContents::ParseMetaData() in file_reader_contents.cpp
	// Not sure but seems HOST_BUFFER is not supported
	int64_t num_bytes;
	file->GetSize(&num_bytes);

	std::vector<uint8_t> byteData(num_bytes);
	CUDF_CALL( read_file_into_buffer(file, num_bytes, byteData.data(), 100, 10) );

	pq_read_arg readerArgs;
	readerArgs.source_type = HOST_BUFFER;
  readerArgs.source = reinterpret_cast<char*>(byteData.data());
  readerArgs.buffer_size = byteData.size();
  readerArgs.use_cols = nullptr;
  // readerArgs.use_cols_len;

	CUDF_CALL( read_parquet_schema(&readerArgs) );

	gdf_columns_out.resize(readerArgs.num_cols_out);
 	for(size_t i = 0; i < gdf_columns_out.size(); i++){
 		gdf_columns_out[i].create_gdf_column(readerArgs.data[i]);
	}

	free(readerArgs.data);
	free(readerArgs.index_col);
}

} /* namespace io */
} /* namespace ral */
