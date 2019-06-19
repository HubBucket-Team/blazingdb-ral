#include "ParserUtil.h"

#include <cudf.h>
#include "Traits/RuntimeTraits.h"

namespace ral {
namespace io {

// Used to create column new indexes that dont include columns that are already loaded
std::vector<size_t> get_column_indices_not_already_loaded(
            const std::vector<size_t> & column_indices_requested, 
            const std::vector<std::string> & column_names,
            std::map<std::string,std::map<std::string, gdf_column_cpp>> & loaded_columns,
            const std::string & user_readable_file_handle) {

	std::vector<size_t> column_indices;
	for(auto column_index : column_indices_requested){
		bool already_parsed_before = false;
		auto file_iter = loaded_columns.find(user_readable_file_handle);
		if (file_iter != loaded_columns.end()){ // we have already parsed this file before
			auto col_iter = loaded_columns[user_readable_file_handle].find(column_names[column_index]);
			if (col_iter != loaded_columns[user_readable_file_handle].end()){ // we have already parsed this column before
				already_parsed_before = true;				
			}
		}
		if (!already_parsed_before)
			column_indices.push_back(column_index);		
	}
    return column_indices;
}


void get_columns_that_were_already_loaded(
    const std::vector<size_t> & column_indices_requested, 
    const std::vector<std::string> & column_names,
    std::map<std::string,std::map<std::string, gdf_column_cpp>> & loaded_columns,
    const std::string & user_readable_file_handle,
    std::vector<gdf_column_cpp> & columns,
    std::vector<gdf_column_cpp> & columns_out) {
    
    int newly_parsed_col_idx = 0;
	for(auto column_index : column_indices_requested){
		bool already_parsed_before = false;
		const std::string column_name = column_names[column_index];
		auto file_iter = loaded_columns.find(user_readable_file_handle);
		if (file_iter != loaded_columns.end()){ // we have already parsed this file before
			auto col_iter = loaded_columns[user_readable_file_handle].find(column_name);
			if (col_iter != loaded_columns[user_readable_file_handle].end()){ // we have already parsed this column before
				already_parsed_before = true;
				columns_out.push_back(loaded_columns[user_readable_file_handle][column_name]);
			}
		}
		if (!already_parsed_before) {
			if (column_name != columns[newly_parsed_col_idx].name()){
				std::cout<<"ERROR: logic error when trying to use already loaded columns in ParquetParser"<<std::endl;
			}
			columns_out.push_back(columns[newly_parsed_col_idx]);
			loaded_columns[user_readable_file_handle][columns[newly_parsed_col_idx].name()] = columns[newly_parsed_col_idx];
			newly_parsed_col_idx++;			
		}
	}
}



std::vector<gdf_column_cpp> create_empty_columns(const std::vector<std::string> & column_names,
                                                const std::vector<std::string> & column_types){

    std::vector<gdf_column_cpp> columns(column_names.size());

    for (size_t i = 0; i < column_names.size(); i++){
        gdf_dtype type = ral::traits::convert_string_dtype(column_types[i]);
        columns[i].create_gdf_column(type, 0, nullptr, gdf_dtype_size(type), column_names[i]);
    }
    return columns;
}

} /* namespace io */
} /* namespace ral */