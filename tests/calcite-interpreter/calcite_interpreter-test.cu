#include <cstdlib>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include <CalciteExpressionParsing.h>
#include <CalciteInterpreter.h>
#include <DataFrame.h>
#include <GDFColumn.cuh>
#include <Utils.cuh>

#include <gdf/gdf.h>

class TestEnvironment : public testing::Environment {
public:
	virtual ~TestEnvironment() {}
	virtual void SetUp() {}

	void TearDown() {
		cudaDeviceReset();
	}
};

struct calcite_interpreter_TEST : public ::testing::Test {

	void SetUp(){

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

		inputs.resize(3);
		inputs[0].create_gdf_column(GDF_INT8, num_values, (void *) input1, 1);
		inputs[1].create_gdf_column(GDF_INT8, num_values, (void *) input2, 1);
		inputs[2].create_gdf_column(GDF_INT8, num_values, (void *) input3, 1);

		input_tables.push_back(inputs);
		input_tables.push_back(inputs);
	}

	void TearDown(){
		for(int i = 0; i < outputs.size(); i++){
			print_column(outputs[i].get_gdf_column());
		}
	}

	gdf_column_cpp left;
	gdf_column_cpp right;
	gdf_column_cpp third;

	std::vector<gdf_column_cpp> inputs;
	
	size_t num_values = 32;

	std::vector<std::vector<gdf_column_cpp> > input_tables;
	std::vector<std::string> table_names={"hr.emps", "hr.sales"};
	std::vector<std::vector<std::string>> column_names={{"x", "y", "z"},{"a", "b", "x"}};
	void * temp_space;

	std::vector<gdf_column_cpp> outputs;
	std::vector<std::string> output_column_names;
};

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
*/
/*TEST_F(calcite_interpreter_TEST, processing_project1) {

	{   //select x from hr.emps
		std::string query = "\
LogicalProject(x=[$0])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);
	}
}*/

TEST_F(calcite_interpreter_TEST, processing_project2) {

	{ //select z > 5 from hr.emps
		std::string query = "\
LogicalProject(EXPR$0=[>($2, 5)])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);
	}
}
	/*{   //select x from hr.emps
		std::string query = "\
LogicalProject(x=[$0])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		std::vector<gdf_column_cpp> outputs;
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

		std::vector<gdf_column_cpp> outputs;
		std::vector<std::string> output_column_names;

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 2);
	}*/

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

int main(int argc, char **argv){
	::testing::InitGoogleTest(&argc, argv);
	::testing::Environment* const env = ::testing::AddGlobalTestEnvironment(new TestEnvironment());
	return RUN_ALL_TESTS();
}