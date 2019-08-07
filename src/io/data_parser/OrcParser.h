#ifndef ORCPARSER_H_
#define ORCPARSER_H_

#include "DataParser.h"
#include <vector>
#include <memory>
#include "arrow/io/interfaces.h"
#include "GDFColumn.cuh"

namespace ral {
namespace io {

class orc_parser: public data_parser {
public:
	orc_parser();
	virtual ~orc_parser();
	void parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
				const std::string & user_readable_file_handle,
				std::vector<gdf_column_cpp> & columns_out,
				const Schema & schema,
				std::vector<size_t> column_indices_requested);

	void parse_schema(std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > files,
			Schema & schema);


};

} /* namespace io */
} /* namespace ral */

#endif /* ORCPARSER_H_ */
