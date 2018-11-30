/*
 * CSVParser.cpp
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#include "CSVParser.h"

namespace ral {
namespace io {

csv_parser::csv_parser() {
	// TODO Auto-generated constructor stub

}

csv_parser::~csv_parser() {
	// TODO Auto-generated destructor stub
}

gdf_error csv_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			std::vector<gdf_column_cpp> & columns,
			std::vector<bool> include_column){

	gdf_error error;
	//cry becuase i cant call the api
	return error;
}

} /* namespace io */
} /* namespace ral */
