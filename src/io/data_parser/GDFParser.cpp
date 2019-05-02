/*
 * GDFParser.cpp
 *
 *  Created on: Apr 30, 2019
 *      Author: felipe
 */

#include "GDFParser.h"

namespace ral {
namespace io {

gdf_parser::gdf_parser(blazingdb::message::io::FileSystemBlazingTableSchema table_schema) : table_schema(table_schema) {
	// TODO Auto-generated constructor stub

}

gdf_parser::~gdf_parser() {
	// TODO Auto-generated destructor stub
}


void gdf_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
		std::vector<gdf_column_cpp> & columns,
		Schema schema,
		std::vector<size_t> column_indices){
	//call to blazing frame here
}

void gdf_parser::parse_schema(std::shared_ptr<arrow::io::RandomAccessFile> file,
		ral::io::Schema & schema){

	//generate schema from message here
}

}
}
