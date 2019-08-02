/*
 * Schema.h
 *
 *  Created on: Apr 26, 2019
 *      Author: felipe
 */

#ifndef BLAZING_RAL_SCHEMA_H_
#define BLAZING_RAL_SCHEMA_H_


#include <vector>
#include <string>
#include <cudf.h>
#include "../GDFColumn.cuh"
#include <blazingdb/protocol/message/interpreter/utils.h>


namespace ral {
namespace io {

/**
 * I did not want to write this and its very dangerous
 * but the cudf::io::csv::reader_options (what a name) currently requires a char * input
 *I have no idea why
 */
std::string convert_dtype_to_string(const gdf_dtype & dtype);

class Schema {
public:
	Schema();
	Schema(blazingdb::protocol::TableSchemaSTL schema);
	Schema(	std::vector<std::string> names,
			std::vector<size_t> calcite_to_file_indices,
			std::vector<gdf_dtype> types,
			std::vector<size_t> num_row_groups);

	Schema(	std::vector<std::string> names,
			std::vector<gdf_dtype> types);

	virtual ~Schema();

	std::vector<std::string> get_names() const;
	std::vector<std::string> get_types() const;
	std::vector<gdf_dtype> get_dtypes() const;
	std::string get_name(size_t schema_index) const;
	std::string get_type(size_t schema_index) const;
	size_t get_file_index(size_t schema_index) const;
	size_t get_num_row_groups(size_t file_index) const;

	size_t get_num_columns() const;

	void add_column(gdf_column_cpp column,size_t file_index);
	void add_column(std::string name, gdf_dtype type,size_t file_index);

	inline bool operator==(const Schema& rhs) const{
		return (this->names == rhs.names) && (this->types == rhs.types);
	}

	blazingdb::protocol::TableSchemaSTL getTransport();
	inline bool operator!=( const Schema& rhs){ return !(*this == rhs); }
private:
	std::vector<std::string> names;
	std::vector<size_t> calcite_to_file_indices; //maps calcite columns to our columns
	std::vector<gdf_dtype> types;
	std::vector<size_t> num_row_groups;
	int header;
};

} /* namespace io */
} /* namespace ral */

#endif /* BLAZING_RAL_SCHEMA_H_ */
