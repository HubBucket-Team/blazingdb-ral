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

void parquet_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file, std::vector<gdf_column_cpp> & columns) {
	size_t num_row_groups;
	size_t num_cols;
	std::vector< gdf_dtype> dtypes;
	std::vector< std::string> column_names;

 	std::vector<bool> 	include_columns;
	gdf_error error = gdf::parquet::read_schema(file, num_row_groups, num_cols, dtypes, column_names, include_columns);
	this->parse(file, columns, include_columns);
}

void parquet_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			std::vector<gdf_column_cpp> & gdf_columns_out,
			std::vector<bool> include_columns){

	gdf_error error;
	size_t num_row_groups;
	size_t num_cols;
	std::vector<gdf_dtype> dtypes;
	std::vector< std::string> column_names;

	// TODO: The return value of this function is just a placeholder,
	// doesn't return a meaningful value 
	std::vector<bool> 	include_columns_test;
	error = gdf::parquet::read_schema(file, num_row_groups, num_cols, dtypes, column_names, include_columns_test);

	std::vector<std::size_t> column_indices;
	for (size_t index =0; index < include_columns.size(); index++) {
		if (include_columns[index]){
			column_indices.push_back(index);	
		}
	}

	std::vector<std::size_t> row_group_ind(num_row_groups); // check, include all row groups
    std::iota(row_group_ind.begin(), row_group_ind.end(), 0);

    std::vector<gdf_column *> columns_out;

	// TODO: Fix this error handling
	error = gdf::parquet::read_parquet_by_ids(file, row_group_ind, column_indices, columns_out);
	if (error != GDF_SUCCESS) {
		throw std::runtime_error("In parquet_parser::parse: error in gdf::parquet::read_parquet_by_ids");
	}
	
	auto n_cols = columns_out.size();
	gdf_columns_out.resize(n_cols);

 	for(size_t i = 0; i < n_cols; i++ ){
	    gdf_column	*column = columns_out[i];
		column->col_name = nullptr;
 		gdf_columns_out[i].create_gdf_column(column);
		gdf_columns_out[i].set_name(column_names[ column_indices[i] ]);
	}
}

void parquet_parser::parse_schema(std::shared_ptr<arrow::io::RandomAccessFile> file, std::vector<gdf_column_cpp> & gdf_columns_out)  {
	size_t num_row_groups;
	size_t num_cols;
	std::vector<gdf_dtype> dtypes;
	std::vector< std::string> column_names;

	// TODO: The return value of this function is just a placeholder,
	// doesn't return a meaningful value 
	std::vector<bool> 	include_columns;
	gdf_error error = gdf::parquet::read_schema(file, num_row_groups, num_cols, dtypes, column_names, include_columns);
 
	std::vector<std::size_t> row_group_ind(num_row_groups); // check, include all row groups
    std::iota(row_group_ind.begin(), row_group_ind.end(), 0);

	auto n_cols = column_names.size();
	gdf_columns_out.resize(n_cols);

	for (size_t i = 0; i < dtypes.size(); i++) {
		auto dtype = dtypes[i];
		gdf_columns_out[i].create_gdf_column(dtype, 0U, nullptr, 0U, column_names[i]);
	}
}

} /* namespace io */
} /* namespace ral */
