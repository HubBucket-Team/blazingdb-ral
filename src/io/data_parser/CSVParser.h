/*
 * CSVParser.h
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#ifndef CSVPARSER_H_
#define CSVPARSER_H_

#include "DataParser.h"
#include <vector>
#include <memory>
#include "arrow/io/interfaces.h"
#include "GDFColumn.cuh"

namespace ral {
namespace io {

class csv_parser: public data_parser {
public:
	csv_parser();
	virtual ~csv_parser();
	gdf_error parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
				std::vector<gdf_column_cpp> & columns,
				std::vector<bool> include_column);
};

} /* namespace io */
} /* namespace ral */

#endif /* CSVPARSER_H_ */
