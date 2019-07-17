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
#include <cudf/io_types.hpp>

namespace ral {
namespace io {

class csv_parser: public data_parser {
public:
	csv_parser(std::string  delimiter,
			 std::string  line_terminator,
			int skip_rows,
			 std::vector<std::string>  names,
			std::vector<gdf_dtype>  dtypes);
	csv_parser(cudf::io::csv::reader_options	args);

	virtual ~csv_parser();


	void parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			const std::string & user_readable_file_handle,
			std::vector<gdf_column_cpp> & columns_out,
			const Schema & schema,
			std::vector<size_t> column_indices_requested);

	void parse_schema(std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > files,
			ral::io::Schema & schema);

private:
	char quote_character = '\"';
	cudf::io::csv::reader_options args;
	std::vector<std::string> column_names;
	std::vector<std::string> dtype_strings; //this is only because we have to convert for args and dont want to have to remember to free up all the junk later	
};

} /* namespace io */
} /* namespace ral */

#endif /* CSVPARSER_H_ */
