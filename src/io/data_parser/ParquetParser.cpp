/*
 * ParquetParser.cpp
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#include "ParquetParser.h"
#include "cudf/io_functions.hpp"

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

void parquet_parser::parse_schema(std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > files, ral::io::Schema & schema)  {

	gdf_error error = gdf::parquet::read_schema(files, schema);
}

} /* namespace io */
} /* namespace ral */
