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

//@todo, replace with new c++14
int dtype_size(gdf_dtype col_type) {
  switch( col_type )
    {
    case GDF_INT8:
      {
        using ColType = int8_t;

        return sizeof(ColType);
      }
    case GDF_INT16:
      {
        using ColType = int16_t;

        return sizeof(ColType);
      }
    case GDF_INT32:
      {
        using ColType = int32_t;

        return sizeof(ColType);
      }
    case GDF_INT64:
      {
        using ColType = int64_t;

        return sizeof(ColType);
      }
    case GDF_FLOAT32:
      {
        using ColType = float;

        return sizeof(ColType);
      }
    case GDF_FLOAT64:
      {
        using ColType = double;

        return sizeof(ColType);
      }

    default:
      assert( false );//type not handled
    }
    return 0;
}


gdf_column_cpp ToGdfColumnCpp(const std::string &name,
                              const gdf_dtype    dtype,
                              const std::size_t  length,
                              const void *       data,
                              const std::size_t  size) {
  gdf_column_cpp column_cpp;
  column_cpp.create_gdf_column(dtype, length, const_cast<void *>(data), size);
  column_cpp.delete_set_name(name);
  return column_cpp;
}



gdf_error parquet_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			std::vector<gdf_column_cpp> & gdfColumnsCpps,
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
	size_t index = 0;
	for (auto column : columns_out) {
	    size_t type_size = dtype_size(column->dtype);
    	gdfColumnsCpps.push_back(ToGdfColumnCpp(std::to_string(index) + "_col", column->dtype, column->size, column->data, type_size));
		index++;
	}
	return error;
}

} /* namespace io */
} /* namespace ral */
