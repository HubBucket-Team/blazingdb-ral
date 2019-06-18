

#ifndef GDFPARSER_H_
#define GDFPARSER_H_

#include "DataParser.h"
#include <vector>
#include <memory>
#include "arrow/io/interfaces.h"
#include <blazingdb/protocol/message/io/file_system.h>
#include "GDFColumn.cuh"
#include "cudf.h"


namespace ral {
namespace io {

class gdf_parser: public data_parser {
public:
	gdf_parser(blazingdb::message::io::FileSystemBlazingTableSchema table_schema, uint64_t accessToken);

	virtual ~gdf_parser();


	void parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
			const std::string & user_readable_file_handle,
			std::vector<gdf_column_cpp> & columns,
			const Schema & schema,
			std::vector<size_t> column_indices);


	void parse_schema(std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > files,
			ral::io::Schema & schema);

;

private:
	blazingdb::message::io::FileSystemBlazingTableSchema table_schema;
	std::vector<void *> handles;
	uint64_t access_token;
	std::map<std::string, gdf_column_cpp> loaded_columns;
};

} /* namespace io */
} /* namespace ral */

#endif /* GDFPARSER_H_ */
