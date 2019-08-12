/*
 * jsonParser.cpp
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#include "JSONParser.h"
#include <cudf/io_functions.hpp>
#include <cudf/legacy/column.hpp>
#include <blazingdb/io/Util/StringUtil.h>

#include <arrow/io/file.h>
#include <arrow/status.h>

#include <thread>

#include <GDFColumn.cuh>
#include <GDFCounter.cuh>

#include "../Schema.h"
#include "io/data_parser/ParserUtil.h"

#include <numeric>

namespace ral {
namespace io {

json_parser::json_parser() {
	// TODO Auto-generated constructor stub

}

json_parser::~json_parser() {
	// TODO Auto-generated destructor stub
}

/**
 * @brief read in a JSON file
 *
 * Read in a JSON file, extract all fields, and return a GDF (array of gdf_columns) using arrow interface
 **/

cudf::table read_json_arrow(std::shared_ptr<arrow::io::RandomAccessFile> arrow_file_handle, bool first_row_only = false)
{
	int64_t 	num_bytes;
	arrow_file_handle->GetSize(&num_bytes);

	if (first_row_only && num_bytes > 8192) // lets only read up to 8192 bytes. We are assuming that a full row will always be less than that
		num_bytes = 8192;

	std::string buffer;
	buffer.resize(num_bytes);

	gdf_error error = read_file_into_buffer(arrow_file_handle, num_bytes, (uint8_t*) (buffer.c_str()),100,10);
	assert(error == GDF_SUCCESS);
	
	cudf::json_read_arg args(cudf::source_info(buffer.c_str(), num_bytes));
	args.lines = true; //TODO hardcoded

	cudf::table table_out;

	if (first_row_only) {
		args.byte_range_offset = 0;
		args.byte_range_size = num_bytes;
	}

	table_out = cudf::read_json(args);
	
	arrow_file_handle->Close();

	return table_out;
}

void json_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
	const std::string & user_readable_file_handle,
	std::vector<gdf_column_cpp> & columns_out,
	const Schema & schema,
	std::vector<size_t> column_indices){

	if (column_indices.size() == 0){ // including all columns by default
		column_indices.resize(schema.get_num_columns());
		std::iota(column_indices.begin(), column_indices.end(), 0);
	}

	if (file == nullptr){
		columns_out = create_empty_columns(schema.get_names(), schema.get_dtypes(), column_indices);
		return;
	}
	
	if (column_indices.size() > 0){
		cudf::table table_out;
		table_out = read_json_arrow(file);

		assert(table_out.num_columns() > 0);

		columns_out.resize(column_indices.size());
		for(size_t i = 0; i < columns_out.size(); i++){
			if (table_out.get_column(i)->dtype == GDF_STRING){
				NVStrings* strs = static_cast<NVStrings*>(table_out.get_column(column_indices[i])->data);
				NVCategory* category = NVCategory::create_from_strings(*strs);
				std::string column_name(table_out.get_column(column_indices[i])->col_name);
				columns_out[i].create_gdf_column(category, table_out.get_column(column_indices[i])->size, column_name);
				gdf_column_free(table_out.get_column(column_indices[i]));
			} else {
				columns_out[i].create_gdf_column(table_out.get_column(column_indices[i]));
			}
		}
	}
}

void json_parser::parse_schema(const std::string & user_readable_file_handle, std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > files, ral::io::Schema & schema_out)  {

	cudf::table table_out;
	table_out = read_json_arrow(files[0], true);

	assert(table_out.num_columns() > 0);

 	for(size_t i = 0; i < table_out.num_columns(); i++ ){
		gdf_column_cpp c;
		c.create_gdf_column(table_out.get_column(i));
		c.set_name(table_out.get_column(i)->col_name);
		schema_out.add_column(c,i);
	 }
}

} /* namespace io */
} /* namespace ral */
