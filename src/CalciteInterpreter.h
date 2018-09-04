
#ifndef CALCITEINTERPRETER_H_
#define CALCITEINTERPRETER_H_

#include <iostream>
#include <vector>
#include <gdf/gdf.h>
#include <string>
#include "CalciteExpressionParsing.h"
#include "DataFrame.h"

gdf_error evaluate_query(
		std::vector<std::vector<gdf_column *> > input_tables,
		std::vector<std::string> table_names,
		std::vector<std::vector<std::string>> column_names,
		std::string query,
		std::vector<gdf_column *> outputs,
		std::vector<std::string> output_column_names,
		void * temp_space);

blazing_frame evalute_split_query(
		std::vector<std::vector<gdf_column *> > input_tables,
		std::vector<std::string> table_names,
		std::vector<std::vector<std::string>> column_names,
		std::vector<std::string> query);

std::string get_named_expression(std::string query_part, std::string expression_name);

#endif /* CALCITEINTERPRETER_H_ */
