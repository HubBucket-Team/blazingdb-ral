/*
 * DataParser.h
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#ifndef DATAPARSER_H_
#define DATAPARSER_H_

#include <vector>
#include <memory>
#include "arrow/io/interfaces.h"
#include "GDFColumn.cuh"

namespace ral {
namespace io {

class data_parser {
public:
	/**
	 * columns should be the full size of the schema, if for example, some of the columns
	 * are not oing to be parsed, we will still want a gdf_column_cpp of size 0
	 * in there so we can preserve column index like access e.g. $3 $1 from the logical plan
	 *
	 */
	virtual gdf_error parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			std::vector<gdf_column_cpp> & columns,
			std::vector<bool> include_column) = 0;

	virtual gdf_error parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			std::vector<gdf_column_cpp> & columns) = 0;

	virtual gdf_error parse_schema(std::shared_ptr<arrow::io::RandomAccessFile> file,
			std::vector<gdf_column_cpp> & columns) = 0;

};

} /* namespace io */
} /* namespace ral */

#endif /* DATAPARSER_H_ */
