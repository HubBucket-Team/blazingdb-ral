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
#include <iostream>
#include "../Utils.cuh"

#define checkError(error, txt)  if ( error != GDF_SUCCESS) { std::cerr << "ERROR:  " << error <<  "  in "  << txt << std::endl;  return error; }

namespace ral {
namespace io {

void init_default_csv_args(csv_read_arg & args){

	args.delimiter = '|';
	args.lineterminator = '\n';
	args.quotechar = '"';
	args.quoting = QUOTE_MINIMAL;
	args.doublequote = true;
	args.delim_whitespace = false;
	args.skipinitialspace = false;
	args.dayfirst = false;
	args.skiprows = 0;
	args.skipfooter = 0;
	args.mangle_dupe_cols = true;
	args.windowslinetermination = false;
	args.compression = nullptr;
	args.decimal = '.';
	// args.thousands
	args.skip_blank_lines = true;
	// args.comment
	args.keep_default_na = true;
	args.na_filter = false;
	// args.prefix
	args.nrows = -1;
	args.header = -1;
}

void copy_non_data_csv_args(csv_read_arg & args, csv_read_arg & new_args){
	new_args.num_cols		= args.num_cols;
	new_args.names			= args.names;
	new_args.dtype			= args.dtype;
	new_args.delimiter		= args.delimiter;
	new_args.lineterminator = args.lineterminator;
	new_args.skip_blank_lines = args.skip_blank_lines;
	new_args.header 		= args.header;
	new_args.nrows 			= args.nrows;
	new_args.decimal 		= args.decimal;
	new_args.quotechar 		= args.quotechar;
	new_args.quoting 		= args.quoting;
	new_args.doublequote 	= args.doublequote;
	new_args.delim_whitespace = args.delim_whitespace;
	new_args.skipinitialspace = args.skipinitialspace;
	new_args.dayfirst 		= args.dayfirst;
	new_args.skiprows 		= args.skiprows;
	new_args.skipfooter 	= args.skipfooter;
	new_args.mangle_dupe_cols = args.mangle_dupe_cols;
	new_args.windowslinetermination	= args.windowslinetermination;
	new_args.compression 	= args.compression;
	new_args.keep_default_na = args.keep_default_na;
	new_args.na_filter		= args.na_filter;

}

/**
 * I did not want to write this and its very dangerous
 * but the csv_read_arg (what a name) currently requires a char * input
 *I have no idea why
 */
std::string convert_dtype_to_string(const gdf_dtype & dtype) {

	if(dtype == GDF_STRING)			return "str";
	if(dtype == GDF_DATE64)			return "date64";
	if(dtype == GDF_DATE32)			return "date32";
	if(dtype == GDF_TIMESTAMP)		return "timestamp";
	if(dtype == GDF_CATEGORY)		return "category";
	if(dtype == GDF_FLOAT32)		return "float32";
	if(dtype == GDF_FLOAT64)		return "float64";
	if(dtype == GDF_INT16)			return "short";
	if(dtype == GDF_INT32)			return "int32";
	if(dtype == GDF_INT64)			return "int64";

	return "str";
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

gdf_error read_csv_arrow(csv_read_arg *args, std::shared_ptr<arrow::io::RandomAccessFile> arrow_file_handle)
{
	void * 		map_data = NULL;
	int64_t 	num_bytes;
	arrow_file_handle->GetSize(&num_bytes);
	map_data = (void *) malloc(num_bytes);
	gdf_error error = read_file_into_buffer(arrow_file_handle, num_bytes, (uint8_t*) map_data,100,10);
	checkError(error, "reading from file into system memory");

	args->input_data_form = gdf_csv_input_form::HOST_BUFFER;
	args->filepath_or_buffer = (const char *)map_data;
	args->buffer_size = num_bytes;

	error = read_csv(args);
	free(map_data);

	//done reading data from map
	arrow_file_handle->Close();

	return error;
}


csv_parser::csv_parser(const std::string & delimiter,
		const std::string & line_terminator,
		int skip_rows,
		const std::vector<std::string> & names,
		const std::vector<gdf_dtype> & dtypes) {

	init_default_csv_args(args);
	args.delimiter 		= delimiter[0];
	args.lineterminator = line_terminator[0];
	this->column_names = names;
	this->dtype_strings.resize(dtypes.size());
	for(int i = 0; i < dtypes.size(); i++){
		this->dtype_strings[i] = convert_dtype_to_string(dtypes[i]);
	}
}


csv_parser::csv_parser(csv_read_arg args) {
	this->args = args;
}

csv_parser::~csv_parser() {

}

//schema is not really necessary yet here, but we want it to maintain compatibility
void csv_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
		std::vector<gdf_column_cpp> & columns,
		Schema schema,
		std::vector<size_t> column_indices){

	// TODO this function needs to be revisited. the cudf csv reader now supports actually selecting what columns you want

	csv_read_arg raw_args{};

	args.num_cols = schema.get_names().size();
	if(this->column_names.size() > 0){
		args.names = new const char *[num_columns];
		for(int column_index = 0; column_index < num_columns; column_index++){
			args.names[column_index] = this->column_names[column_index].c_str();
		}
	}else{
		args.names = nullptr;
	}

	if(this->dtype_strings.size() > 0){
		args.dtype = new const char *[num_columns]; //because dynamically allocating metadata is fun
		for(int column_index = 0; column_index < num_columns; column_index++){

			args.dtype[column_index] = this->dtype_strings[column_index].c_str();
		}

	}else{
		args.dtype = nullptr;
	}

	std::vector<int> column_indices_int(column_indices.size());
	for(int i = 0; i < column_indices.size(); i++){
		column_indices_int[i] = column_indices[i];
	}

	if(column_indices.size() > 0){
		args.use_cols_int = column_indices_int.data();
		args.ues_cols_int_len = column_indices_int.size();
	}

	copy_non_data_csv_args(args, raw_args);

	CUDF_CALL(read_csv_arrow(&raw_args,file));

	//	std::cout << "args.num_cols_out " << raw_args.num_cols_out << std::endl;
	//	std::cout << "args.num_rows_out " <<raw_args.num_rows_out << std::endl;
	assert(raw_args.num_cols_out > 0);

	for(size_t i = 0; i < raw_args.num_cols_out; i++ ){
		gdf_column_cpp c;
		c.create_gdf_column(raw_args.data[i]);
		columns.push_back(c);
	}

	delete []args.dtype;
	args.dtype = nullptr;
	delete []args.names;
	args.names = nullptr;
}

void csv_parser::parse_schema(std::shared_ptr<arrow::io::RandomAccessFile> file, ral::io::Schema & schema)  {
	csv_read_arg raw_args{};
	copy_non_data_csv_args(args, raw_args);

	std::shared_ptr< arrow::io::Buffer > buffer;
	arrow::AllocateBuffer(8192, &buffer);
	file->ReadAt(0,8192,&buffer);
	auto buffer_reader = std::make_shared<arrow::io::BufferReader>(buffer);

	raw_args.nrows=1;
	CUDF_CALL(read_csv_arrow(&raw_args,buffer_reader));
	buffer_reader->Close();
	//	std::cout << "args.num_cols_out " << raw_args.num_cols_out << std::endl;
	//	std::cout << "args.num_rows_out " <<raw_args.num_rows_out << std::endl;
	assert(raw_args.num_cols_out > 0);

	for(size_t i = 0; i < raw_args.num_cols_out; i++ ){
		gdf_column_cpp c;
		c.create_gdf_column(raw_args.data[i]); 
		c.set_name(std::string{raw_args.names[i]});
		schema.add_column(c,i);

	}
}

} /* namespace io */
} /* namespace ral */
