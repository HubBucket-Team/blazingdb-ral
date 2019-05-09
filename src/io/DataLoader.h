/*
 * dataloader.h
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#pragma once

#include <FileSystem/Uri.h>

#include <vector>
#include "GDFColumn.cuh"
#include "data_parser/DataParser.h"
#include "data_provider/DataProvider.h"
#include <arrow/io/interfaces.h>
#include <memory>
#include "Schema.h"

namespace ral {
namespace io {
/**
 * class is used for loading data from some kind of file type using some kind of file provider
 * in our case we will be using blazing-io to read and write files but a local version could also be made
 */

class data_loader {
public:
	data_loader(data_parser  * parser, data_provider * provider);
	virtual ~data_loader();

	/**
	 * loads data into a vector of gdf_column_cpp
	 * @param columns a vector to receive our output should be of size 0 when it is coming in and it will be allocated by this function
	 * @param include_column the different files we can read from can have more columns than we actual want to read, this lest us filter some of them out
	 */
	void load_data(std::vector<gdf_column_cpp> & columns, const std::vector<size_t> & column_indices, const Schema & schema);
	void get_schema(Schema & schema);
private:
	/**
	 * DataProviders are able to serve up one or more arrow::io::RandomAccessFile objects
	 */
	data_provider * provider;
	/**
	 * parsers are able to parse arrow::io::RandomAccessFile objects of a specific file type and convert them into gdf_column_cpp
	 */
	data_parser * parser;
};



} /* namespace io */
} /* namespace ral */

