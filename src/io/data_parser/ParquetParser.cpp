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

	std::vector<std::size_t> column_indices_mapped_to_parquet; //this handles the situation where sometimes not all columns were supported, calcite only knows about supported columns

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

    std::vector<gdf_column *> columns_out;

	// TODO: Fix this error handling
	error = gdf::parquet::read_parquet_by_ids(file, row_group_ind, column_indices_mapped_to_parquet, columns_out);
	if (error != GDF_SUCCESS) {
		throw std::runtime_error("In parquet_parser::parse: error in gdf::parquet::read_parquet_by_ids");
	}
	

}

void parquet_parser::parse_schema(std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > files, ral::io::Schema & schema)  {

	gdf_error error = gdf::parquet::read_schema(files, schema);
 

}

} /* namespace io */
} /* namespace ral */
