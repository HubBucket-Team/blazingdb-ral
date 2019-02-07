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

void parquet_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file, std::vector<gdf_column_cpp> & columns) {
	size_t num_row_groups;
	size_t num_cols;
	std::vector< ::parquet::Type::type> parquet_dtypes;
	std::vector< std::string> column_names;

	// TODO: The return value of this function is just a placeholder,
	// doesn't return a meaningful value 
	gdf_error error = gdf::parquet::read_schema(file, num_row_groups, num_cols, parquet_dtypes, column_names);

 	std::vector<bool> 	include_column;

#define WHEN(TYPE)                                  \
    case ::parquet::Type::TYPE:                     \
        include_column.push_back(true);				\
        break

	for (size_t i = 0; i < parquet_dtypes.size(); i++) {
		  switch (parquet_dtypes[i]) {
			WHEN(BOOLEAN);
            WHEN(INT32);
            WHEN(INT64);
            WHEN(FLOAT);
            WHEN(DOUBLE);
			default:
		        include_column.push_back(false);
				std::cerr << parquet_dtypes[i] << " - Column type not supported" << std::endl;
				throw std::runtime_error("In parquet_parser::parse: column type not supported");
		  }
	}

#undef WHEN

	this->parse(file, columns, include_column);
}

void parquet_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			std::vector<gdf_column_cpp> & gdf_columns_out,
			std::vector<bool> include_column){

	gdf_error error;
	size_t num_row_groups;
	size_t num_cols;
	std::vector< ::parquet::Type::type> parquet_dtypes;
	std::vector< std::string> column_names;

	// TODO: The return value of this function is just a placeholder,
	// doesn't return a meaningful value 
	error = gdf::parquet::read_schema(file, num_row_groups, num_cols, parquet_dtypes, column_names);	

	std::vector<std::size_t> column_indices;
	for (size_t index =0; index < include_column.size(); index++) {
		if (include_column[index]){
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
		gdf_columns_out[i].delete_set_name(column_names[ column_indices[i] ]);
	}
}

void parquet_parser::parse_schema(std::shared_ptr<arrow::io::RandomAccessFile> file, std::vector<gdf_column_cpp> & gdf_columns_out)  {
	size_t num_row_groups;
	size_t num_cols;
	std::vector< ::parquet::Type::type> parquet_dtypes;
	std::vector< std::string> column_names;

	// TODO: The return value of this function is just a placeholder,
	// doesn't return a meaningful value 
	gdf_error error = gdf::parquet::read_schema(file, num_row_groups, num_cols, parquet_dtypes, column_names);
 
	std::vector<std::size_t> row_group_ind(num_row_groups); // check, include all row groups
    std::iota(row_group_ind.begin(), row_group_ind.end(), 0);

	auto n_cols = column_names.size();
	gdf_columns_out.resize(n_cols);

	for (size_t i = 0; i < parquet_dtypes.size(); i++) {
		switch (parquet_dtypes[i]) {
		#define WHEN(dtype, TYPE)                          												        \
				case ::parquet::Type::TYPE:                    											   	    \
				gdf_columns_out[i].create_gdf_column(dtype, 0U, nullptr, 0U, column_names[i]);		  			\
					break
			WHEN(GDF_INT8, BOOLEAN);
            WHEN(GDF_INT32, INT32);
            WHEN(GDF_INT64, INT64);
            WHEN(GDF_FLOAT32, FLOAT);
            WHEN(GDF_FLOAT64, DOUBLE);
			default:
				std::cerr << parquet_dtypes[i] << " - Column type not supported" << std::endl;
				throw std::runtime_error("In parquet_parser::parse_schema: column type not supported");
			#undef WHEN
		}
	}
}

} /* namespace io */
} /* namespace ral */
