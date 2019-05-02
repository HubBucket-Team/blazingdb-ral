/*
 * Schema.cpp
 *
 *  Created on: Apr 26, 2019
 *      Author: felipe
 */

#include "Schema.h"

namespace ral {
namespace io {

Schema::Schema(	std::vector<std::string> names,
		std::vector<size_t> calcite_to_file_indices,
		std::vector<gdf_dtype> types,
		size_t num_row_groups) : names(names),calcite_to_file_indices(calcite_to_file_indices), types(types), num_row_groups(num_row_groups) {
	// TODO Auto-generated constructor stub

}

Schema::Schema(	std::vector<std::string> names,
		std::vector<gdf_dtype> types) : names(names),calcite_to_file_indices({}), types(types), num_row_groups(1) {


}

Schema::~Schema() {
	// TODO Auto-generated destructor stub
}

std::vector<std::string> Schema::get_names(){
	return this->names;
}
std::vector<std::string> Schema::get_types(){
	return this->types;
}
std::string Schema::get_name(size_t schema_index){
	return this->names[schema_index];
}
std::string Schema::get_type(size_t schema_index){
	return this->types[schema_index];
}

size_t Schema::get_file_index(size_t schema_index){
	if(this->calcite_to_file_indices.size() == 0){
		return schema_index;
	}
	return this->calcite_to_file_indices(index);
}

size_t Schema::get_num_row_groups(){
	return this->num_row_groups;
}

void Schema::add_column(gdf_column_cpp column,size_t file_index){
	this->names.push_back(column.name());
	this->types.push_back(column.dtype());
	this->calcite_to_file_indices.push_back(file_index);
}

void Schema::add_column(std::string name, gdf_dtype type,size_t file_index){
	this->names.push_back(name);
	this->types.push_back(type);
	this->calcite_to_file_indices.push_back(file_index);
}

} /* namespace io */
} /* namespace ral */
