/*
 * CSVParser.cpp
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#include "CSVParser.h"
#include "cudf/io_types.h"
#include <arrow/status.h>
#include <arrow/io/interfaces.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <arrow/buffer.h>
#include <iostream>
#include "../Utils.cuh"
#include "io/data_parser/ParserUtil.h"

#include <algorithm>

#define checkError(error, txt)  if ( error != GDF_SUCCESS) { std::cerr << "ERROR:  " << error <<  "  in "  << txt << std::endl;  return error; }

namespace ral {
namespace io {

void init_default_csv_args(cudf::io::csv::reader_options & args){

	args.delimiter = '|';
	args.lineterminator = '\n';
	args.quotechar = '"';
	args.quoting = cudf::io::csv::quote_style::QUOTE_MINIMAL;
	args.doublequote = false;
	args.delim_whitespace = false;
	args.skipinitialspace = false;
	args.dayfirst = false;
	args.mangle_dupe_cols = true;
	args.compression = "none";
	args.decimal = '.';
	// args.thousands
	args.skip_blank_lines = true;
	// args.comment
	args.keep_default_na = true;
	args.na_filter = false;
	// args.prefix
	args.header = -1;	
}

void copy_non_data_csv_args(cudf::io::csv::reader_options & args, cudf::io::csv::reader_options & new_args){
	new_args.names			= args.names;
    new_args.dtype			= args.dtype;
    new_args.delimiter		= args.delimiter;
    new_args.lineterminator = args.lineterminator;
	new_args.skip_blank_lines = args.skip_blank_lines;
	new_args.header 		= args.header;
	new_args.decimal 		= args.decimal;
	new_args.quotechar 		= args.quotechar;
	new_args.quoting 		= args.quoting;
	new_args.doublequote 	= args.doublequote;
	new_args.delim_whitespace = args.delim_whitespace;
	new_args.skipinitialspace = args.skipinitialspace;
	new_args.dayfirst 		= args.dayfirst;
	new_args.mangle_dupe_cols = args.mangle_dupe_cols;
	new_args.compression 	= args.compression;
	new_args.keep_default_na = args.keep_default_na;
	new_args.na_filter		= args.na_filter;
	new_args.use_cols_indexes = args.use_cols_indexes;
	
}



/**
 * reads contents of an arrow::io::RandomAccessFile in a char * buffer up to the number of bytes specified in bytes_to_read
 * for non local filesystems where latency and availability can be an issue it will ret`ry until it has exhausted its the read attemps and empty reads that are allowed
 */
gdf_error read_file_into_buffer(std::shared_ptr<arrow::io::RandomAccessFile> file, int64_t bytes_to_read, uint8_t* buffer, int total_read_attempts_allowed, int empty_reads_allowed){

	if (bytes_to_read > 0){

		int64_t total_read;
		arrow::Status status = file->Read(bytes_to_read,&total_read, buffer);

		if (!status.ok()){
			return GDF_FILE_ERROR;
		}

		if (total_read < bytes_to_read){
			//the following two variables shoudl be explained
			//Certain file systems can timeout like hdfs or nfs,
			//so we shoudl introduce the capacity to retry
			int total_read_attempts = 0;
			int empty_reads = 0;

			while (total_read < bytes_to_read && total_read_attempts < total_read_attempts_allowed && empty_reads < empty_reads_allowed){
				int64_t bytes_read;
				status = file->Read(bytes_to_read-total_read,&bytes_read, buffer + total_read);
				if (!status.ok()){
					return GDF_FILE_ERROR;
				}
				if (bytes_read == 0){
					empty_reads++;
				}
				total_read += bytes_read;
			}
			if (total_read < bytes_to_read){
				return GDF_FILE_ERROR;
			} else {
				return GDF_SUCCESS;
			}
		} else {
			return GDF_SUCCESS;
		}
	} else {
		return GDF_SUCCESS;
	}
}



/**
 * @brief read in a CSV file
 *
 * Read in a CSV file, extract all fields, and return a GDF (array of gdf_columns) using arrow interface
 **/

cudf::table read_csv_arrow(cudf::io::csv::reader_options args, std::shared_ptr<arrow::io::RandomAccessFile> arrow_file_handle, bool first_row_only = false)
{
	int64_t 	num_bytes;
	arrow_file_handle->GetSize(&num_bytes);

	if (first_row_only && num_bytes > 8192) // lets only read up to 8192 bytes. We are assuming that a full row will always be less than that
		num_bytes = 8192;

	args.filepath_or_buffer.resize(num_bytes);

	gdf_error error = read_file_into_buffer(arrow_file_handle, num_bytes, (uint8_t*) (args.filepath_or_buffer.c_str()),100,10);
	assert(error == GDF_SUCCESS);
	
	cudf::io::csv::reader csv_reader(args);
	cudf::table table_out;
	if (first_row_only)
		table_out = csv_reader.read_rows(0, 0, 1);
	else 
		table_out = csv_reader.read_byte_range(0, num_bytes);
	
	arrow_file_handle->Close();
	args.filepath_or_buffer.resize(0);

	return table_out;
}


csv_parser::csv_parser(std::string delimiter,
		 std::string line_terminator,
		int skip_rows,
		std::vector<std::string> names,
		std::vector<gdf_dtype> dtypes) {

	init_default_csv_args(args);

	args.delimiter 		= delimiter[0];
	args.lineterminator = line_terminator[0];
	this->column_names = names;
	this->dtype_strings.resize(dtypes.size());
	for(int i = 0; i < dtypes.size(); i++){
		this->dtype_strings[i] = convert_dtype_to_string(dtypes[i]);
	}
}


csv_parser::csv_parser(cudf::io::csv::reader_options args) {
	this->args = args;
}

csv_parser::~csv_parser() {

}

//schema is not really necessary yet here, but we want it to maintain compatibility
void csv_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
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

		cudf::io::csv::reader_options raw_args{};

		args.names = this->column_names;
		args.dtype = this->dtype_strings;

		// copy column_indices into use_col_indexes
		std::copy(column_indices.begin(), column_indices.end(), 
          std::back_inserter(args.use_cols_indexes));
		
		copy_non_data_csv_args(args, raw_args);

		cudf::table table_out = read_csv_arrow(raw_args,file);

		assert(table_out.num_columns() > 0);

		//column_indices may be requested in a specific order (not necessarily sorted), but read_csv will output the columns in the sorted order, so we need to put them back into the order we want
		std::vector<size_t> idx(column_indices.size());
		std::iota(idx.begin(), idx.end(), 0);
		// sort indexes based on comparing values in column_indices
		std::sort(idx.begin(), idx.end(),
		[&column_indices](size_t i1, size_t i2) {return column_indices[i1] < column_indices[i2];});

		columns_out.resize(column_indices.size());
		for(size_t i = 0; i < columns_out.size(); i++){

			if (table_out.get_column(i)->dtype == GDF_STRING){
				NVStrings* strs = static_cast<NVStrings*>(table_out.get_column(i)->data);
				NVCategory* category = NVCategory::create_from_strings(*strs);
				std::string column_name(table_out.get_column(i)->col_name);
				columns_out[idx[i]].create_gdf_column(category, table_out.get_column(i)->size, column_name);
				gdf_column_free(table_out.get_column(i));
			} else {
				columns_out[idx[i]].create_gdf_column(table_out.get_column(i));
			}			
		}
	}	
}

void csv_parser::parse_schema(std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > files, ral::io::Schema & schema)  {
	cudf::io::csv::reader_options raw_args{};

	args.names = this->column_names;
	args.dtype = this->dtype_strings;

	copy_non_data_csv_args(args, raw_args);

	cudf::table table_out = read_csv_arrow(raw_args, files[0], true);
	
	assert(table_out.num_columns() > 0);

	for(size_t i = 0; i < table_out.num_columns(); i++ ){
		gdf_column_cpp c;
		c.create_gdf_column(table_out.get_column(i)); 
		c.set_name(args.names[i]);
		schema.add_column(c,i);
	}
}

} /* namespace io */
} /* namespace ral */
