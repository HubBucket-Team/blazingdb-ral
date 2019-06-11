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

	gdf_error error;
	size_t num_row_groups = schema.get_num_row_groups(file_index);
	size_t num_cols = schema.get_names().size();
	std::vector< std::string> column_names = schema.get_names();

	// handles the situation where sometimes not all columns were supported, calcite only knows about supported columns
	std::vector<std::size_t> column_indices_mapped_to_parquet;	// BETTER vector<char*>

	if(column_indices.size() == 0){
		column_indices_mapped_to_parquet.resize(schema.get_names().size());
		for (size_t column_index =0;column_index < schema.get_names().size(); column_index++) {
			column_indices_mapped_to_parquet[column_index] = schema.get_file_index(column_index);
		}
	}else{
		column_indices_mapped_to_parquet.resize(column_indices.size());
		for (size_t column_index =0;column_index < column_indices.size(); column_index++) {
			column_indices_mapped_to_parquet[column_index] = schema.get_file_index(column_indices[column_index]);
		}
	}

	std::vector<std::size_t> row_group_ind(num_row_groups); // check, include all row groups
    std::iota(row_group_ind.begin(), row_group_ind.end(), 0);

    std::vector<gdf_column *> columns_ptr(columns.size());
    for(int i = 0; i < columns.size(); i++){
    	columns_ptr[i] = columns[i].get_gdf_column();
    }

	error = gdf::parquet::read_parquet_by_ids(file, row_group_ind, column_indices_mapped_to_parquet, columns_ptr);

	// Is it necessary?
	for(auto column : columns){
		 if (column.get_gdf_column()->dtype == GDF_STRING){
			 NVStrings* strs = static_cast<NVStrings*>(column.get_gdf_column()->data);
			 NVCategory* category = NVCategory::create_from_strings(*strs);
			 column.get_gdf_column()->data = nullptr;
			 column.create_gdf_column(category, column.size(), column.name());
		 }
	}

	if (error != GDF_SUCCESS) {
		throw std::runtime_error("In parquet_parser::parse: error in gdf::parquet::read_parquet_by_ids");
	}


	// CHANGES
	pq_read_arg* pq_args = new pq_read_arg;

	// Fill data to pq_args
	pq_args->source_type = ARROW_RANDOM_ACCESS_FILE;
	pq_args->strings_to_categorical = false;
	pq_args->row_group = -1;	// Set to read all Row Groups (RG)
	pq_args->skip_rows = 0;		// set to read from the row 0 in a RG
	pq_args->num_rows = -1;		// Set to read until the last row
	pq_args->use_cols_len = static_cast<int>(column_indices_mapped_to_parquet.size());
	pq_args->data = nullptr;

	if(column_indices.size() == 0){
		column_indices_mapped_to_parquet.resize(schema.get_names().size());
		for (size_t column_i = 0; column_i < schema.get_names().size(); column_i++) {
			pq_args->use_cols[column_i] = (char*) schema.get_file_index(column_i);
		}
	} else {
		for (size_t column_i = 0; column_i < column_indices.size(); column_i++) {
			pq_args->use_cols[column_i] = (char*) schema.get_file_index(column_indices[column_i]);
		}
	}

	// Call the new read parquet
	gdf_error error_ = read_parquet_arrow(pq_args, file);

	columns.resize(pq_args->num_cols_out);
	for (int i = 0; i < pq_args->num_cols_out; i++){
		columns[i].create_gdf_column(pq_args->data[i]);
	}

	if (error_ != GDF_SUCCESS) {
		throw std::runtime_error("In parquet_parser::parse: error in read_parquet_arrow");
	}
	
}

void parquet_parser::parse_schema(std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > files, ral::io::Schema & schema)  {

	gdf_error error = gdf::parquet::read_schema(files, schema);
}

} /* namespace io */
} /* namespace ral */
