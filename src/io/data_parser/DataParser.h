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
	
	virtual gdf_error parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			std::vector<gdf_column_cpp> & columns,
			std::vector<bool> include_column) = 0;
private:

};

} /* namespace io */
} /* namespace ral */

#endif /* DATAPARSER_H_ */
