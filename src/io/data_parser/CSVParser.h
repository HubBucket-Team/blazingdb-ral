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
#include "cudf.h"

namespace ral {
namespace io {

class csv_parser: public data_parser {
public:
	csv_parser(std::string  delimiter,
			 std::string  line_terminator,
			int skip_rows,
			 std::vector<std::string>  names,
			std::vector<gdf_dtype>  dtypes);
	csv_parser(csv_read_arg	args);

	virtual ~csv_parser();


	void parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			std::vector<gdf_column_cpp> & columns,
			const Schema & schema,
			std::vector<size_t> column_indices,
			size_t file_index);

	void parse_schema(std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > files,
			ral::io::Schema & schema);

private:
	char quote_character = '\"';
	csv_read_arg args;
	std::vector<std::string> column_names;
	std::vector<std::string> dtype_strings; //this is only because we have to convert for args and dont want to have to remember to free up all the junk later
	std::map<std::string, gdf_column_cpp> loaded_columns;
};

} /* namespace io */
} /* namespace ral */

#endif /* CSVPARSER_H_ */
