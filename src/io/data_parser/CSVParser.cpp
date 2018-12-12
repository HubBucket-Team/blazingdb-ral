/*
 * CSVParser.cpp
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#include "CSVParser.h"
#include "cudf/io_types.h"
#include "cudf/io_functions_cpp.h"

namespace ral {
namespace io {

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
		std::vector<std::string> names,
		std::vector<gdf_dtype> dtypes) {


	args.windowslinetermination = false;
	args.delimiter 		= ',';

	if(names.size() != dtypes.size()){
		//TODO: something graceful, like a swan diving in a pristine lake, not this duck flinging feces in a coop
	}
	int num_columns = names.size();
	if(delimiter.size() == 1){
		args.lineterminator = '\n';
	}else{
		if(delimiter[0] == '\n' && delimiter[1] == '\r'){
			args.lineterminator = '\n';
			args.windowslinetermination = true;
		}else{
			//god knows what to do, set an invalid flag in this class?
			//i guess this is why factories are good
		}
	}
	this->column_names.resize(num_columns);
	this->dtype_strings.resize(num_columns);

	args.names = new const char *[num_columns];
	args.dtype = new const char *[num_columns]; //because dynamically allocating metadata is fun
	for(int column_index = 0; column_index < num_columns; column_index++){
		this->dtype_strings[column_index] = convert_dtype_to_string(dtypes[column_index]);
		args.dtype[column_index] = this->dtype_strings[column_index].c_str();
		args.names[column_index] = this->column_names[column_index].c_str();
	}
	args.delim_whitespace = 0;
	args.skipinitialspace = 0;
	args.skiprows 		= skip_rows;
	args.skipfooter 	= 0;
	args.dayfirst 		= 0;
	args.use_cols_int 	= NULL;
	args.use_cols_char 	= NULL;
	args.use_cols_char_len  = 0;
	args.use_cols_int_len   = 0;
	args.nrows = -1;
	args.parse_dates = true;
	args.encoding = NULL;
	*args.quotechar = '"';


}

csv_parser::csv_parser(){
	//TODO: enable autodetcion
}
csv_parser::csv_parser(csv_read_arg args) {
	this->args = args;
}

csv_parser::~csv_parser() {
	delete args.dtype;
}

gdf_error csv_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
		std::vector<gdf_column_cpp> & columns,
		std::vector<bool> include_column){


	gdf_error error = read_csv_arrow(&this->args,file);


	return error;
}

} /* namespace io */
} /* namespace ral */
