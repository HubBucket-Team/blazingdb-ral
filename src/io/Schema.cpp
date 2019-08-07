/*
 * Schema.cpp
 *
 *  Created on: Apr 26, 2019
 *      Author: felipe
 */

#include "Schema.h"

namespace ral {
namespace io {



std::string convert_dtype_to_string(const gdf_dtype & dtype) {

	if(dtype == GDF_STRING)			return "str";
	if(dtype == GDF_DATE64)			return "date64";
	if(dtype == GDF_DATE32)			return "date32";
	if(dtype == GDF_TIMESTAMP)		return "timestamp";
	if(dtype == GDF_CATEGORY)		return "category";
	if(dtype == GDF_FLOAT32)		return "float32";
	if(dtype == GDF_FLOAT64)		return "float64";
	if(dtype == GDF_INT16)			return "short";
	if(dtype == GDF_INT32)			return "int32";
	if(dtype == GDF_INT64)			return "int64";

	return "str";
}

Schema::Schema(	std::vector<std::string> names,
		std::vector<size_t> calcite_to_file_indices,
		std::vector<gdf_dtype> types,
		std::vector<size_t> num_row_groups) : names(names),calcite_to_file_indices(calcite_to_file_indices), types(types), num_row_groups(num_row_groups) {
	// TODO Auto-generated constructor stub

}

Schema::Schema(	std::vector<std::string> names,
		std::vector<gdf_dtype> types) : names(names),calcite_to_file_indices({}), types(types), num_row_groups({}) {


}

Schema::Schema() : names({}),calcite_to_file_indices({}), types({}), num_row_groups({}) {



}

Schema::Schema(blazingdb::protocol::TableSchemaSTL schema){
	this->calcite_to_file_indices = schema.calciteToFileIndices;
	this->names = schema.names;
	this->types.resize(schema.types.size());
	for(int i = 0; i < schema.types.size(); i++){
		this->types[i] = (gdf_dtype) schema.types[i];
	}
	this->num_row_groups = schema.numRowGroups;

}


Schema::~Schema() {
	// TODO Auto-generated destructor stub
}

std::vector<std::string> Schema::get_names() const{
	return this->names;
}
std::vector<std::string> Schema::get_types() const{
	std::vector<std::string> string_types;
	for(int i = 0; i < this->types.size(); i++){
		string_types.push_back(convert_dtype_to_string(this->types[i]));
	}
	return string_types;
}
std::vector<gdf_dtype> Schema::get_dtypes() const{
	return this->types;		
}
std::string Schema::get_name(size_t schema_index) const{
	return this->names[schema_index];
}
std::string Schema::get_type(size_t schema_index) const{
	return convert_dtype_to_string(this->types[schema_index]);
}

size_t Schema::get_file_index(size_t schema_index) const{
	if(this->calcite_to_file_indices.size() == 0){
		return schema_index;
	}
	return this->calcite_to_file_indices[schema_index];
}

size_t Schema::get_num_row_groups(size_t file_index) const{
	return this->num_row_groups[file_index];
}

size_t Schema::get_num_columns() const {
	return this->names.size();
}

blazingdb::protocol::TableSchemaSTL Schema::getTransport(){
	blazingdb::protocol::TableSchemaSTL transport_schema;

	transport_schema.names = this->names;
	transport_schema.calciteToFileIndices = this->calcite_to_file_indices;
	for(int i = 0; i < this->types.size(); i++){
		transport_schema.types.push_back(this->types[i]);
	}

	transport_schema.numRowGroups = this->num_row_groups;
	return transport_schema;
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
