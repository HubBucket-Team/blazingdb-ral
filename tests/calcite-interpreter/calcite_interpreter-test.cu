#include <cstdlib>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include <CalciteExpressionParsing.h>
#include <CalciteInterpreter.h>
#include <DataFrame.h>
#include <Utils.cuh>

#include <gdf/gdf.h>

TEST(calcite_interpreter_TEST, processing_queries) {

	{
		// Initializating data

		std::vector<std::vector<gdf_column *> > input_tables = {{new gdf_column, new gdf_column, new gdf_column}};
		std::vector<std::string> table_names={"hr.joiner"};
		std::vector<std::vector<std::string>> column_names={{"c1" ,"c2" ,"c3"}};
		std::vector<gdf_column *> outputs;
		std::vector<std::string> output_column_names;
		void * temp_space;


		std::string query = "\
LogicalScan(table=[[hr, joiner]]";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);

		EXPECT_TRUE(outputs.size() == 3);
	}
}
