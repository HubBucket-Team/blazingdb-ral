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

	gdf_column left;
	gdf_column right;
	gdf_column third;

	std::vector<gdf_column *> inputs(3);
	inputs[0] = &left;
	inputs[1] = &right;
	inputs[2] = &third;

	size_t num_values = 32;

	char * input1 = new char[num_values];
	char * input2 = new char[num_values];
	char * input3 = new char[num_values];
	for(int i = 0; i < num_values; i++){
		if(i % 2 == 0){
			input1[i] = 1;
		}else{
			input1[i] = i;
		}
		input2[i] = i;
		input3[i] = 1;
	}

	create_gdf_column(inputs[0], GDF_INT8, num_values, (void *) input1, 1);
	create_gdf_column(inputs[1], GDF_INT8, num_values, (void *) input2, 1);
	create_gdf_column(inputs[2], GDF_INT8, num_values, (void *) input3, 1);

	// Initializating data

	std::vector<std::vector<gdf_column *> > input_tables = {{&left, &right, &third}, {&left, &right, &third}};
	std::vector<std::string> table_names={"hr.emps", "hr.sales"};
	std::vector<std::vector<std::string>> column_names={{"x", "y", "z"},{"a", "b", "x"}};
	void * temp_space;

	/*{   //select * from hr.emps
		std::string query = "\
LogicalProject(x=[$0], y=[$1], z=[$2])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		std::vector<gdf_column *> outputs;
		std::vector<std::string> output_column_names;

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 3);
	}

	{   //select x from hr.emps
		std::string query = "\
LogicalProject(x=[$0])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		std::vector<gdf_column *> outputs;
		std::vector<std::string> output_column_names;

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);
	}*/

	{ //select z > 5 from hr.emps
		std::string query = "\
LogicalProject(EXPR$0=[>($2, 5)])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		std::vector<gdf_column *> outputs;
		std::vector<std::string> output_column_names;

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);
	}

	{   //select x from hr.emps
		std::string query = "\
LogicalProject(x=[$0])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		std::vector<gdf_column *> outputs;
		std::vector<std::string> output_column_names;

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);
	}

	{   //select x + y, z from hr.emps
		std::string query = "\
LogicalProject(EXPR$0=[+($0, $1)], z=[$2])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		std::vector<gdf_column *> outputs;
		std::vector<std::string> output_column_names;

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 2);
	}

	/*{   //select x from hr.emps where y = z
		std::string query = "\
LogicalProject(x=[$0])\n\
  LogicalFilter(condition=[=($1, $2)])\n\
    EnumerableTableScan(table=[[hr, emps]])";

		std::vector<gdf_column *> outputs;
		std::vector<std::string> output_column_names;

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);
	}

	{   //select x + y as S from hr.emps
		std::string query = "\
LogicalProject(S=[+($0, $1)])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		std::vector<gdf_column *> outputs;
		std::vector<std::string> output_column_names;

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);
	}*/
	
}
