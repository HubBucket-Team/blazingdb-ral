/*
 * ParquetParser.cpp
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#include "ParquetParser.h"


#include <parquet/api.h>

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
 
gdf_error parquet_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			std::vector<gdf_column_cpp> & gdf_columns_out,
			std::vector<bool> include_column){

	gdf_error error;
	const std::vector<std::size_t> row_group_indices = {0};//@todo check it

    std::vector<std::size_t> column_indices;
	for (size_t index =0; index < include_column.size(); index++) {
		if (include_column[index]){
			column_indices.push_back(index);	
		}
	}
    std::vector<gdf_column *> columns_out;
	gdf_error error_code = gdf::parquet::read_parquet_by_ids(file, row_group_indices, column_indices, columns_out);	
	auto n_cols = columns_out.size();
	gdf_columns_out.resize(n_cols);

 	for(size_t i = 0; i < n_cols; i++ ){
	    gdf_column	*column = columns_out[i];
		column->col_name = nullptr;
		gdf_columns_out[i].create_gdf_column(column);
	}
	return error;
}

} /* namespace io */
} /* namespace ral */
