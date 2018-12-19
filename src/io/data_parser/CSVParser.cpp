/*
 * CSVParser.cpp
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#include "CSVParser.h"
#include "cudf/io_types.h"
#include "cudf/io_functions_cpp.h"
#include <iostream>
namespace ral {
namespace io {

/**
 * csv reader takes gdf_column** and sets up
 */
void create_csv_args(size_t num_columns,
		csv_read_arg & args,
		const std::vector<bool> & include_column ){

	size_t num_columns_output = 0;
	for(size_t column_index = 0; column_index < num_columns; column_index++){
		if(include_column[0])	num_columns_output++;
	}


	args.data = new gdf_column*[num_columns_output];
	args.use_cols_int = new int[num_columns_output];
	args.use_cols_char_len = num_columns_output;
	size_t output_column_index = 0;
	for(size_t column_index = 0; column_index < num_columns; column_index++){
		if(include_column[column_index]){
			args.data[output_column_index] = new gdf_column;
			args.use_cols_int[output_column_index] = column_index;
			output_column_index++;
		}
	}
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

csv_parser::csv_parser(const std::string & delimiter,
		const std::string & line_terminator,
		int skip_rows,
		const std::vector<std::string> & names,
		const std::vector<gdf_dtype> & dtypes) {


	// args.windowslinetermination = false;
	
	// if(names.size() != dtypes.size()){
	// 	//TODO: something graceful, like a swan diving in a pristine lake, not this duck flinging feces in a coop
	// }
	int num_columns = names.size();
	// if(delimiter.size() == 1){
	// 	args.lineterminator = '\n';
	// }else{
	// 	if(delimiter[0] == '\n' && delimiter[1] == '\r'){
	// 		args.lineterminator = '\n';
	// 		args.windowslinetermination = true;
	// 	}else{
	// 		//god knows what to do, set an invalid flag in this class?
	// 		//i guess this is why factories are good
	// 	}
	// }

	this->column_names = names;
	this->dtype_strings.resize(num_columns);
	std::cout<<"made it here"<<std::endl;
	args.names = new const char *[num_columns];
	args.dtype = new const char *[num_columns]; //because dynamically allocating metadata is fun
	for(int column_index = 0; column_index < num_columns; column_index++){
		this->dtype_strings[column_index] = convert_dtype_to_string(dtypes[column_index]);
		args.dtype[column_index] = this->dtype_strings[column_index].c_str();
		args.names[column_index] = this->column_names[column_index].c_str();
	}
	std::cout<<"made it here too"<<std::endl;
	// args.num_cols = num_columns; //why not?
	// args.delim_whitespace = 0;
	// args.skipinitialspace = 0;
	// args.skiprows 		= skip_rows;
	// args.skipfooter 	= 0;
	// args.dayfirst 		= 0;

	// args.mangle_dupe_cols=true;
	// args.num_cols_out=0;

	// args.use_cols_int       = NULL;
	// args.use_cols_char      = NULL;
	// args.use_cols_char_len  = 0;
	// args.use_cols_int_len   = 0;
	// // args.parse_dates = true; @check
	// std::cout<<"if im last it was me"<<std::endl;
	// args.quotechar = this->quote_character;
	// std::cout<<"nope still good"<<std::endl;

	args.num_cols		= num_columns; 
	args.delimiter 		= delimiter[0];
	args.lineterminator = line_terminator[0];
}


csv_parser::csv_parser(csv_read_arg args) {
	this->args = args;
}

csv_parser::~csv_parser() {
	delete []args.dtype;
	delete []args.names;
}

gdf_error csv_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file, std::vector<gdf_column_cpp> & columns) {

	gdf_error error = read_csv_arrow(&this->args,file);

	for(size_t i = 0; i < args.num_cols_out; i++ ){
		gdf_column_cpp c;
		c.create_gdf_column(args.data[i]); 
		c.delete_set_name(std::string{args.names[i]});
		columns.push_back(c);
	}
}

gdf_error csv_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
		std::vector<gdf_column_cpp> & columns,
		std::vector<bool> include_column){

	// create_csv_args(num_columns,
	// 		args,
	// 		include_column );

	gdf_error error = read_csv_arrow(&this->args,file);

	std::cout << "args.num_cols_out " << args.num_cols_out << std::endl;
	std::cout << "args.num_rows_out " <<args.num_rows_out << std::endl;
	assert(args.num_cols_out > 0);

	//This is kind of legacy but we need to copy the name to gdf_column_cpp
	//TODO: make it so we dont have to do this
	// for(size_t output_column_index = 0; output_column_index < args.use_cols_int_len; output_column_index++){
	// 	size_t index_in_original_columns = args.use_cols_int[output_column_index];
	// 	columns[index_in_original_columns].create_gdf_column(
	// 			args.data[output_column_index]);
	// }
 	for(size_t i = 0; i < args.num_cols_out; i++ ){
		if(include_column[i]) {
			gdf_column_cpp c;
			c.create_gdf_column(args.data[i]); 
			c.delete_set_name(std::string{args.names[i]});
			columns.push_back(c);
		}
	}
	return error;
}
gdf_error csv_parser::parse_schema(std::shared_ptr<arrow::io::RandomAccessFile> file, std::vector<gdf_column_cpp> & gdf_columns_out)  {
	gdf_error error;

	return error;
}
} /* namespace io */
} /* namespace ral */
