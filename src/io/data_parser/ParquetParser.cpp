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
		Schema schema,
		std::vector<size_t> column_indices){

    Schema current_schema;
    this->parse_schema(file,current_schema);
    if(current_schema != schema){
    	throw std::exception("schema mismatch on parquet files");
    }

	gdf_error error;
	size_t num_row_groups = current_schema.get_num_row_groups();
	size_t num_cols = current_schema.get_names().size();
	std::vector<gdf_dtype> dtypes = current_schema.get_types();
	std::vector< std::string> column_names = current_schema.get_names();


	std::vector<std::size_t> column_indices_mapped_to_parquet(column_indices.size()); //this handles the situation where sometimes not all columns were supported, calcite only knows about supported columns

	for (size_t column_index =0;column_index < column_indices.size(); column_index++) {
		column_indices_mapped_to_parquet[column_index] = current_schema.get_file_index(column_index);
	}

	std::vector<std::size_t> row_group_ind(num_row_groups); // check, include all row groups
    std::iota(row_group_ind.begin(), row_group_ind.end(), 0);

    std::vector<gdf_column *> columns_out;

	// TODO: Fix this error handling
	error = gdf::parquet::read_parquet_by_ids(file, row_group_ind, column_indices_mapped_to_parquet, columns_out);
	if (error != GDF_SUCCESS) {
		throw std::runtime_error("In parquet_parser::parse: error in gdf::parquet::read_parquet_by_ids");
	}
	
	auto n_cols = columns_out.size();
	gdf_columns_out.resize(n_cols);

 	for(size_t i = 0; i < n_cols; i++ ){
	    gdf_column	*column = columns_out[i];
 		gdf_columns_out[i].create_gdf_column(column);
	}
}

void parquet_parser::parse_schema(std::shared_ptr<arrow::io::RandomAccessFile> file, ral::io::Schema & schema)  {
	size_t num_row_groups;
	size_t num_cols;
	std::vector<gdf_dtype> dtypes;
	std::vector< std::string> column_names;

	// TODO: The return value of this function is just a placeholder,
	// doesn't return a meaningful value 
	std::vector<bool> 	include_columns;
	gdf_error error = gdf::parquet::read_schema(file, schema);
 

	size_t n_cols = 0;
	for (size_t index =0; index < include_columns.size(); index++) {
		if (include_columns[index]){
			n_cols++;
		}
	}
	gdf_columns_out.resize(n_cols);
	size_t index =0;
	for (size_t i = 0; i < dtypes.size(); i++) {
		if (include_columns[i]){
			auto dtype = dtypes[i];
			gdf_columns_out[index].create_gdf_column(dtype, 0U, nullptr, 0U, column_names[i]);
			schema.add_column(column_names[i],dtype,i);
			index++;
		}
	}
}

} /* namespace io */
} /* namespace ral */
