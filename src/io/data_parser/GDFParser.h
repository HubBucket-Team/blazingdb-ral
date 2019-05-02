

#ifndef GDFPARSER_H_
#define GDFPARSER_H_

#include "DataParser.h"
#include <vector>
#include <memory>
#include "arrow/io/interfaces.h"
#include "GDFColumn.cuh"
#include "cudf.h"

namespace ral {
namespace io {

class gdf_parser: public data_parser {
public:
	gdf_parser(blazingdb::message::io::FileSystemBlazingTableSchema table_schema);

	virtual ~gdf_parser();


	void parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			std::vector<gdf_column_cpp> & columns,
			Schema schema,
			std::vector<size_t> column_indices);

	void parse_schema(std::shared_ptr<arrow::io::RandomAccessFile> file,
			ral::io::Schema & schema);

private:
	blazingdb::message::io::FileSystemBlazingTableSchema table_schema;
};

} /* namespace io */
} /* namespace ral */

#endif /* GDFPARSER_H_ */
