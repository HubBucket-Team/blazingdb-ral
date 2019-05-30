/*
 * Schema.h
 *
 *  Created on: Apr 26, 2019
 *      Author: felipe
 */

#ifndef SCHEMA_H_
#define SCHEMA_H_


#include <vector>
#include <string>
#include <cudf.h>
#include "../GDFColumn.cuh"

namespace ral {
namespace io {

/**
 * I did not want to write this and its very dangerous
 * but the csv_read_arg (what a name) currently requires a char * input
 *I have no idea why
 */
std::string convert_dtype_to_string(const gdf_dtype & dtype);

class Schema {
public:
	Schema();
	Schema(	std::vector<std::string> names,
			std::vector<size_t> calcite_to_file_indices,
			std::vector<gdf_dtype> types,
			std::vector<size_t> num_row_groups);

	Schema(	std::vector<std::string> names,
			std::vector<gdf_dtype> types);

	virtual ~Schema();

	std::vector<std::string> get_names() const;
	std::vector<std::string> get_types() const;
	std::string get_name(size_t schema_index) const;
	std::string get_type(size_t schema_index) const;
	size_t get_file_index(size_t schema_index) const;
	size_t get_num_row_groups(size_t file_index) const;

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

};

} /* namespace io */
} /* namespace ral */

#endif /* SCHEMA_H_ */
