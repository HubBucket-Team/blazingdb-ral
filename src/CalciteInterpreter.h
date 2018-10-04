
#ifndef CALCITEINTERPRETER_H_
#define CALCITEINTERPRETER_H_

#include <iostream>
#include <vector>
#include <string>
#include "DataFrame.h"
#include "Types.h"
#include <gdf/gdf.h>

blazing_frame evalute_split_query(
		std::vector<std::vector<gdf_column_cpp> > input_tables,
		std::vector<std::string> table_names,
		std::vector<std::vector<std::string>> column_names,
		std::vector<std::string> query);

query_token_t evaluate_query(
		std::vector<std::vector<gdf_column_cpp> > input_tables,
		std::vector<std::string> table_names,
		std::vector<std::vector<std::string>> column_names,
		std::string logicalPlan,
		connection_id_t connection);

std::string get_named_expression(std::string query_part, std::string expression_name);

#endif /* CALCITEINTERPRETER_H_ */
