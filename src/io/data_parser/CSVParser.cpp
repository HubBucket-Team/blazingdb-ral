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

void initil_default_values(cudf::csv_read_arg & args) {
	// Todo: Almost all params are already set
	args.source.type = HOST_BUFFER;
	//args.skipfooter = 0;
}

void copy_non_data_csv_read_args(cudf::csv_read_arg & args, cudf::csv_read_arg & new_args){
	// Todo: Review which more need to be added
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
	new_args.use_cols_names = args.use_cols_names;
	new_args.source.type = args.source.type;
	new_args.source.filepath = args.source.filepath;
	new_args.source.file = args.source.file;
	new_args.source.buffer.first = args.source.buffer.first;
	new_args.source.buffer.second = args.source.buffer.second;
	new_args.skiprows = args.skiprows;
	new_args.nrows = args.nrows;
	new_args.skipfooter = args.skipfooter;
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


cudf::table read_csv_arg_arrow(cudf::csv_read_arg args, std::shared_ptr<arrow::io::RandomAccessFile> arrow_file_handle, bool first_row_only = false) {
	int64_t 	num_bytes;
	arrow_file_handle->GetSize(&num_bytes);
	
	// lets only read up to 8192 bytes. We are assuming that a full row will always be less than that
	if (first_row_only && num_bytes > 8192) num_bytes = 8192;

	args.source.buffer.second = num_bytes;
	args.source.buffer.first = (const char *) malloc(num_bytes);
	
	gdf_error error = read_file_into_buffer(arrow_file_handle, num_bytes, (uint8_t*) args.source.buffer.first,100,10);
	assert(error == GDF_SUCCESS);

	cudf::table table_out = read_csv(args);

	arrow_file_handle->Close();
	args.source.buffer.second = 0;
	return table_out;
}


csv_parser::csv_parser(std::string delimiter,
		std::string lineterminator,
		int skiprows,
		int header,
		std::vector<std::string> names,
		std::vector<gdf_dtype> dtypes) {

	initil_default_values(csv_arg);

	csv_arg.delimiter = delimiter[0];
	csv_arg.lineterminator = lineterminator[0];
	csv_arg.skiprows = skiprows;
	csv_arg.header = header;		
	
	this->column_names = names;
	this->dtype_strings.resize(dtypes.size());
	for(int i = 0; i < dtypes.size(); i++){
		this->dtype_strings[i] = convert_dtype_to_string(dtypes[i]);
	}
}


csv_parser::csv_parser(cudf::csv_read_arg args) {
	this->csv_arg = args;
}


csv_parser::~csv_parser() {

}

//schema is not really necessary yet here, but we want it to maintain compatibility
void csv_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
		const std::string & user_readable_file_handle,
		std::vector<gdf_column_cpp> & columns_out,
		const Schema & schema,
		std::vector<size_t> column_indices) {
	
	// including all columns by default
	if (column_indices.size() == 0) {
		column_indices.resize(schema.get_num_columns());
		std::iota(column_indices.begin(), column_indices.end(), 0);
	}

	if (file == nullptr) {
		columns_out = create_empty_columns(schema.get_names(), schema.get_dtypes(), column_indices);
		return;
	}

	if (column_indices.size() > 0) {

		cudf::csv_read_arg raw_args = cudf::csv_read_arg{ cudf::source_info{""} };
		csv_arg.names = this->column_names;
		csv_arg.dtype = this->dtype_strings;

		// copy column_indices into use_col_indexes (at the moment is ordered only)
		csv_arg.use_cols_indexes.resize(column_indices.size());
		csv_arg.use_cols_indexes.assign(column_indices.begin(), column_indices.end());
		
		copy_non_data_csv_read_args(csv_arg, raw_args);
		cudf::table table_out = read_csv_arg_arrow(raw_args, file);

		assert(table_out.num_columns() > 0);

		std::vector<size_t> idx(column_indices.size());
		std::iota(idx.begin(), idx.end(), 0);	
		
		// sort indexes based on comparing values in column_indices
		std::sort(idx.begin(), idx.end(),
		[&column_indices](size_t i1, size_t i2) {return column_indices[i1] < column_indices[i2];});

		columns_out.resize(column_indices.size());
		for (size_t i = 0; i < columns_out.size(); i++){
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

void csv_parser::parse_schema(std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > files, ral::io::Schema & schema) {

	cudf::csv_read_arg raw_args = cudf::csv_read_arg{ cudf::source_info{""} };
	csv_arg.names = this->column_names;
	csv_arg.dtype = this->dtype_strings;

	copy_non_data_csv_read_args(csv_arg, raw_args);

	cudf::table table_out = read_csv_arg_arrow(raw_args, files[0], true);
	
	assert(table_out.num_columns() > 0);

	for (size_t i = 0; i < table_out.num_columns(); i++){
		gdf_column_cpp c;
		c.create_gdf_column(table_out.get_column(i)); 
		if ( i < csv_arg.names.size() ) c.set_name(csv_arg.names[i]);
		schema.add_column(c,i);
	}
}

} /* namespace io */
} /* namespace ral */