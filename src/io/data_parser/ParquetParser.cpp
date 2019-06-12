/*
 * ParquetParser.cpp
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#include "ParquetParser.h"
#include "cudf/io_functions.hpp"
#include <blazingdb/io/Util/StringUtil.h>

#include <arrow/io/file.h>
#include <parquet/file_reader.h>
#include <parquet/schema.h>

#include <GDFColumn.cuh>
#include <GDFCounter.cuh>

#include "../Schema.h"


namespace ral {
namespace io {

parquet_parser::parquet_parser() {
	// TODO Auto-generated constructor stub

}

parquet_parser::~parquet_parser() {
	// TODO Auto-generated destructor stub
}

void parquet_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
		std::vector<gdf_column_cpp> & columns,
		const Schema & schema,
		std::vector<size_t> column_indices,
		size_t file_index){

	if (column_indices.size() == 0){ // including all columns by default
		column_indices.resize(schema.get_num_columns());
		std::iota(column_indices.begin(), column_indices.end(), 0);
	}

	// CHANGES
	pq_read_arg pq_args;

	// Fill data to pq_args
	pq_args.source_type = ARROW_RANDOM_ACCESS_FILE;
	pq_args.strings_to_categorical = false;
	pq_args.row_group = -1;	// Set to read all Row Groups (RG)
	pq_args.skip_rows = 0;		// set to read from the row 0 in a RG
	pq_args.num_rows = -1;		// Set to read until the last row
	pq_args.use_cols_len = static_cast<int>(column_indices.size());
	pq_args.use_cols = new const char*[column_indices.size()];
	
	for (size_t column_i = 0; column_i < column_indices.size(); column_i++) {
		std::string col_name = schema.get_name(column_indices[column_i]);
		col_name.push_back(0);
		pq_args.use_cols[column_i] = new char[col_name.size()+1];
		std::memcpy((void *) pq_args.use_cols[column_i], col_name.c_str(), col_name.size());
	}
	
	gdf_error error_;
	try {
		// Call the new read parquet
		error_ = read_parquet_arrow(&pq_args, file);
	} catch (...) {
		for (size_t column_i = 0; column_i < column_indices.size(); column_i++) {
			delete [] pq_args.use_cols[column_i];
		}
		delete [] pq_args.use_cols;
	}
	for (size_t column_i = 0; column_i < column_indices.size(); column_i++) {
		delete [] pq_args.use_cols[column_i];
	}
	delete [] pq_args.use_cols;

	if (error_ != GDF_SUCCESS) {
		throw std::runtime_error("In parquet_parser::parse: error in read_parquet_arrow");
	}

	std::vector<gdf_column_cpp> tmp_columns(pq_args.num_cols_out);
	for(size_t i = 0; i < pq_args.num_cols_out; i++ ){
		tmp_columns[i].create_gdf_column(pq_args.data[i]);
	}

	columns.resize(pq_args.num_cols_out);
	for(size_t i = 0; i < columns.size(); i++){
		 if (tmp_columns[i].get_gdf_column()->dtype == GDF_STRING){
			 NVStrings* strs = static_cast<NVStrings*>(tmp_columns[i].get_gdf_column()->data);
			 NVCategory* category = NVCategory::create_from_strings(*strs);
			 columns[i].create_gdf_column(category, tmp_columns[i].size(), tmp_columns[i].name());
		 } else {
			 columns[i] = tmp_columns[i];
		 }
	}
}

// This function is copied and adapted from cudf
constexpr std::pair<gdf_dtype, gdf_dtype_extra_info> to_dtype(
    parquet::Type::type physical, parquet::LogicalType::type logical) {

	bool strings_to_categorical = false; // parameter used in cudf::read_parquet

  // Logical type used for actual data interpretation; the legacy converted type
  // is superceded by 'logical' type whenever available.
  switch (logical) {
    case parquet::LogicalType::type::UINT_8:
    case parquet::LogicalType::type::INT_8:
      return std::make_pair(GDF_INT8, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::LogicalType::type::UINT_16:
    case parquet::LogicalType::type::INT_16:
      return std::make_pair(GDF_INT16, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::LogicalType::type::DATE:
      return std::make_pair(GDF_DATE32, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::LogicalType::type::TIMESTAMP_MILLIS:
      return std::make_pair(GDF_DATE64, gdf_dtype_extra_info{TIME_UNIT_ms});
    case parquet::LogicalType::type::TIMESTAMP_MICROS:
      return std::make_pair(GDF_DATE64, gdf_dtype_extra_info{TIME_UNIT_us});
    default:
      break;
  }

  // Physical storage type supported by Parquet; controls the on-disk storage
  // format in combination with the encoding type.
  switch (physical) {
    case parquet::Type::type::BOOLEAN:
      return std::make_pair(GDF_BOOL8, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::Type::type::INT32:
      return std::make_pair(GDF_INT32, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::Type::type::INT64:
      return std::make_pair(GDF_INT64, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::Type::type::FLOAT:
      return std::make_pair(GDF_FLOAT32, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::Type::type::DOUBLE:
      return std::make_pair(GDF_FLOAT64, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::Type::type::BYTE_ARRAY:
    case parquet::Type::type::FIXED_LEN_BYTE_ARRAY:
      // Can be mapped to GDF_CATEGORY (32-bit hash) or GDF_STRING (nvstring)
      return std::make_pair(strings_to_categorical ? GDF_CATEGORY : GDF_STRING,
                            gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::Type::type::INT96:
      // Convert Spark INT96 timestamp to GDF_DATE64
      return std::make_pair(GDF_DATE64, gdf_dtype_extra_info{TIME_UNIT_ms});
    default:
      break;
  }

  return std::make_pair(GDF_invalid, gdf_dtype_extra_info{TIME_UNIT_NONE});
}

void parquet_parser::parse_schema(std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > files, ral::io::Schema & schema_out)  {

	// gdf_error error = gdf::parquet::read_schema(files, schema);
	std::vector<std::string> column_names;
	std::vector<gdf_dtype> dtypes;
	std::vector<size_t> num_row_groups(files.size());
	
	std::unique_ptr<parquet::ParquetFileReader> parquet_reader = parquet::ParquetFileReader::Open(files[0]);
	std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
	int total_columns = file_metadata->num_columns();
	num_row_groups[0] = file_metadata->num_row_groups();
	const parquet::SchemaDescriptor * schema = file_metadata->schema();
	for(int i = 0; i < total_columns; i++){
		const parquet::ColumnDescriptor * column = schema->Column(i);
		column_names.push_back(StringUtil::toLower(column->name()));
		std::pair<gdf_dtype, gdf_dtype_extra_info> dtype_pair = to_dtype(column->physical_type(), column->logical_type());
		dtypes.push_back(dtype_pair.first);
	}
	parquet_reader->Close();

	for(int file_index = 1; file_index < files.size(); file_index++){
		parquet_reader = parquet::ParquetFileReader::Open(files[file_index]);
		file_metadata = parquet_reader->metadata();
		const parquet::SchemaDescriptor * schema = file_metadata->schema();
		num_row_groups[file_index] = file_metadata->num_row_groups();
		parquet_reader->Close();
	}

	std::vector<std::size_t> column_indices(total_columns);
    std::iota(column_indices.begin(), column_indices.end(), 0);
	schema_out = ral::io::Schema(column_names,column_indices,dtypes,num_row_groups);
}

} /* namespace io */
} /* namespace ral */
