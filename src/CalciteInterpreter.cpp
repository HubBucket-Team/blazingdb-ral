#include "CalciteInterpreter.h"
#include "StringUtil.h"

gdf_error evaluate_query(
		std::vector<std::vector<gdf_column *> > input_tables,
		std::vector<std::string> table_names,
		std::vector<std::vector<std::string>> column_names,
		std::string query,
		std::vector<gdf_column *> outputs,
		std::vector<std::string> output_column_names,
		void * temp_space){
		
		std::vector<std::string> splitted = StringUtil::split(query, '\n');

		for(auto str : splitted)
			std::cout<<StringUtil::rtrim(str)<<"\n";
}
