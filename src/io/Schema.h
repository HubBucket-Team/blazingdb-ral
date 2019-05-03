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

namespace ral {
namespace io {

class Schema {
public:
	Schema(	std::vector<std::string> names,
			std::vector<size_t> calcite_to_file_indices,
			std::vector<gdf_dtype> types);
	Schema(	std::vector<std::string> names,
			std::vector<gdf_dtype> types);

	virtual ~Schema();

	std::vector<std::string> get_names();
	std::vector<std::string> get_types();
	std::string get_name(size_t schema_index);
	std::string get_type(size_t schema_index);
	size_t get_file_index(size_t schema_index);
	size_t get_num_row_groups();

	void add_column(gdf_column_cpp column,size_t file_index);
	void add_column(std::string name, gdf_dtype type,size_t file_index);

	inline bool operator==(const Schema& lhs, const Schema& rhs){
		return (lhs.names == rhs.names) && (lhs.types == rhs.types);
	}
	inline bool operator!=(const Schema& lhs, const Schema& rhs){ return !(lhs == rhs); }
private:
	std::vector<std::string> names;
	std::vector<size_t> calcite_to_file_indices; //maps calcite columns to our columns
	std::vector<gdf_dtype> types;
	std::vector<size_t> num_row_groups;

};

} /* namespace io */
} /* namespace ral */

#endif /* SCHEMA_H_ */
