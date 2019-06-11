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

	
	std::vector<std::size_t> row_group_ind(num_row_groups); // check, include all row groups
    std::iota(row_group_ind.begin(), row_group_ind.end(), 0);

    std::vector<gdf_column *> columns_ptr(columns.size());
    for(int i = 0; i < columns.size(); i++){
    	columns_ptr[i] = columns[i].get_gdf_column();
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
		pq_args.use_cols[column_i] = schema.get_name(column_indices[column_i]).c_str();
	}

	
	gdf_error error_;
	try {
		// Call the new read parquet
		error_ = read_parquet_arrow(&pq_args, file);
	} catch (...) {
		delete [] pq_args.use_cols;
	}
	delete [] pq_args.use_cols;

	if (error_ != GDF_SUCCESS) {
		throw std::runtime_error("In parquet_parser::parse: error in read_parquet_arrow");
	}

	columns.resize(pq_args.num_cols_out);
	for (int i = 0; i < pq_args.num_cols_out; i++){
		columns[i].create_gdf_column(pq_args.data[i]);
	}

	// // Is it necessary?
	// for(auto column : columns){
	// 	 if (column.get_gdf_column()->dtype == GDF_STRING){
	// 		 NVStrings* strs = static_cast<NVStrings*>(column.get_gdf_column()->data);
	// 		 NVCategory* category = NVCategory::create_from_strings(*strs);
	// 		 column.get_gdf_column()->data = nullptr;
	// 		 column.create_gdf_column(category, column.size(), column.name());
	// 	 }
	// }

	
	
}

void parquet_parser::parse_schema(std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > files, ral::io::Schema & schema)  {

	gdf_error error = gdf::parquet::read_schema(files, schema);
}

} /* namespace io */
} /* namespace ral */
