/*
 * dataloader.h
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#ifndef DATALOADER_H_
#define DATALOADER_H_

#include <FileSystem/Uri.h>

#include <vector>
#include "GDFColumn.cuh"

namespace ral {
namespace io {
/**
 * class is used for loading data from some kind of file type using some kind of file provider
 * in our case we will be using blazing-io to read and write files but a local version could also be made
 */

template <typename DataProvider, typename FileParser>
class data_loader {
public:
	data_loader(FileParser parser, DataProvider data_provider);
	virtual ~data_loader();

	/**
	 * loads data into a vector of gdf_column_cpp
	 * @param columns a vector to receive our output should be of size 0 when it is coming in and it will be allocated by this function
	 * @param include_column the different files we can read from can have more columns than we actual want to read, this lest us filter some of them out
	 */
	void load_data(std::vector<gdf_column_cpp> & columns, std::vector<bool> include_column);
private:
	/**
	 * DataProviders are able to serve up one or more arrow::io::RandomAccessFile objects
	 */
	DataProvider data_provider;
	/**
	 * parsers are able to parse arrow::io::RandomAccessFile objects of a specific file type and convert them into gdf_column_cpp
	 */
	FileParser parser;
};

} /* namespace io */
} /* namespace ral */

#endif /* DATALOADER_H_ */
