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
	csv_parser(const std::string & delimiter,
			const std::string & line_terminator,
			int skip_rows,std::vector<std::string> names,
			std::vector<gdf_dtype> dtypes);
	csv_parser(csv_read_arg	args);
	csv_parser();
	virtual ~csv_parser();
	gdf_error parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
				std::vector<gdf_column_cpp> & columns,
				std::vector<bool> include_column);
private:
	csv_read_arg args;
	std::vector<std::string> column_names;
	std::vector<std::string> dtype_strings; //this is only because we have to convert for args and dont want to have to remember to free up all the junk later
};

} /* namespace io */
} /* namespace ral */

#endif /* CSVPARSER_H_ */
