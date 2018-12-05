/*
 * ParquetParser.cpp
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#include "ParquetParser.h"

namespace ral {
namespace io {

parquet_parser::parquet_parser() {
	// TODO Auto-generated constructor stub

}

parquet_parser::~parquet_parser() {
	// TODO Auto-generated destructor stub
}


gdf_error parquet_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			std::vector<gdf_column_cpp> & columns,
			std::vector<bool> include_column){

	gdf_error error;
	//cry becuase i cant call the api
	return error;
}

} /* namespace io */
} /* namespace ral */
