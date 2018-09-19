#include <cstdlib>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include <CalciteExpressionParsing.h>
#include <CalciteInterpreter.h>
#include <DataFrame.h>
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

		/*inputs.push_back(&left);
		inputs.push_back(&right);
		inputs.push_back(&third);*/

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

		/*create_gdf_column(inputs[0], GDF_INT8, num_values, (void *) input1, 1);
		create_gdf_column(inputs[1], GDF_INT8, num_values, (void *) input2, 1);
		create_gdf_column(inputs[2], GDF_INT8, num_values, (void *) input3, 1);*/

		inputs.resize(3);
		inputs[0] = create_gdf_column(GDF_INT8, num_values, (void *) input1, 1);
		inputs[1] = create_gdf_column(GDF_INT8, num_values, (void *) input2, 1);
		inputs[2] = create_gdf_column(GDF_INT8, num_values, (void *) input3, 1);

		raw_inputs.resize(3);
		raw_inputs[0] = inputs[0].get();
		raw_inputs[1] = inputs[1].get();
		raw_inputs[2] = inputs[2].get();

		input_tables.push_back(raw_inputs);
		input_tables.push_back(raw_inputs);
		/*std::cout<<"[0]\n";
		print_column(&left);
		std::cout<<"[1]\n";
		print_column(&right);
		std::cout<<"[2]\n";
		print_column(&third);
		std::cout<<"----------------\n";*/
	}

	void TearDown(){
		for(int i = 0; i < outputs.size(); i++){
			print_column(outputs[i]);
		}
	}

	gdf_column left;
	gdf_column right;
	gdf_column third;

	std::vector<gdf_col_pointer> inputs;
	std::vector<gdf_column *> raw_inputs;

	size_t num_values = 32;

	void * temp_space;

	std::vector<std::vector<gdf_column *> > input_tables;// = {{&left, &right, &third}, {&left, &right, &third}};
	std::vector<std::string> table_names={"hr.emps", "hr.sales"};
	std::vector<std::vector<std::string>> column_names={{"x", "y", "z"},{"a", "b", "x"}};

	std::vector<gdf_column *> outputs;
	std::vector<std::string> output_column_names;
};


TEST_F(calcite_interpreter_TEST, processing_project0) {

	{   //select * from hr.emps
		std::string query = "\
LogicalProject(x=[$0], y=[$1], z=[$2])\n\
  EnumerableTableScan(table=[[hr, emps]])";
	
		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 3);
	}
}

TEST_F(calcite_interpreter_TEST, processing_project1) {

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
}

/*TEST_F(calcite_interpreter_TEST, processing_project2) {

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

TEST_F(calcite_interpreter_TEST, processing_project3) {

	{ //select a from hr.sales
		std::string query = "\
LogicalProject(a=[$0])\n\
  EnumerableTableScan(table=[[hr, sales]])";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);
	}
}*/

/*TEST_F(calcite_interpreter_TEST, processing_project4) {

	{ //select x + y, z from hr.emps
		std::string query = "\
LogicalProject(EXPR$0=[+($0, $1)], z=[$2])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 2);
	}
}*/

/*TEST_F(calcite_interpreter_TEST, processing_project_filter) {

	{ //select x from hr.emps where y = z
		std::string query = "\
LogicalProject(x=[$0])\n\
  LogicalFilter(condition=[=($1, $2)])\n\
    EnumerableTableScan(table=[[hr, emps]])";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);
	}
}

TEST_F(calcite_interpreter_TEST, processing_project5) {

	{ //select x + y as S from hr.emps
		std::string query = "\
LogicalProject(S=[+($0, $1)])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		//std::cout<<input_tables[0][0]->size<<"#"<<std::endl;
		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);

		//std::cout<<outputs[0]->size<<"="<<std::endl;
		//for(int i = 0; i < outputs.size(); i++){
		//	print_column(outputs[i]);
		//}
		//std::cout<<input_tables[0][0]->size<<"#"<<std::endl;
	}
}

TEST(calcite_interpreter_join_TEST, processing_joins) {
	{
		//lets make a simple test where we have three talbes that we join then filter then project
		//this mimics our java code
		std::vector<std::vector<gdf_column *> > input_tables(2);

		std::vector<gdf_column * > hr_emps(3);
		std::vector<gdf_column * > hr_joiner_1(2);
		//std::vector<gdf_column * > hr_joiner_2(2);

		int emps_x[3] = { 1, 2, 3};
		int emps_y[3] = { 4, 5,6};
		int emps_z[3] = { 10, 10, 10};

		hr_emps[0] = new gdf_column;
		create_gdf_column(hr_emps[0], GDF_INT32, 3, (void *) emps_x, 4);
		hr_emps[1] = new gdf_column;
		create_gdf_column(hr_emps[1], GDF_INT32, 3, (void *) emps_y, 4);
		hr_emps[2] = new gdf_column;
		create_gdf_column(hr_emps[2], GDF_INT32, 3, (void *) emps_z, 4);

		int joiner_join_x[6] = { 1, 1, 1, 2, 2, 3};
		int joiner_y[6] = { 1, 2, 3, 4 ,5 ,6};

		hr_joiner_1[0] = new gdf_column;
		create_gdf_column(hr_joiner_1[0], GDF_INT32, 6, (void *) joiner_join_x, 4);

		hr_joiner_1[1] = new gdf_column;
		create_gdf_column(hr_joiner_1[1], GDF_INT32, 6, (void *) joiner_y, 4);

		input_tables[0] = hr_emps;
		input_tables[1] = hr_joiner_1;
		std::vector<std::string> table_names = { "hr.emps" , "hr.joiner"};
		std::vector<std::vector<std::string>> column_names = {{"x","y","z"},{"join_x","join_y"}};

		std::vector<gdf_column *> outputs;
		std::vector<std::string> output_column_names;
		void * temp_space = nullptr; //we arent really using this it seems

		std::string query = "\
	LogicalProject(x=[$0], y=[$1], z=[$2], join_x=[$3], y0=[$4], EXPR$6=[+($0, $4)])\n\
	LogicalFilter(condition=[OR(<($0, 5), >($3, 3))])\n\
		LogicalJoin(condition=[OR(=($3, $0), =($3, $1))], joinType=[inner])\n\
		EnumerableTableScan(table=[[hr, emps]])\n\
		EnumerableTableScan(table=[[hr, joiner]]) ";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs, output_column_names, temp_space);

		for(int i = 0; i < outputs.size(); i++){
			print_column(outputs[i]);
		}
	}
}*/

int main(int argc, char **argv){
	::testing::InitGoogleTest(&argc, argv);
	::testing::Environment* const env = ::testing::AddGlobalTestEnvironment(new TestEnvironment());
	return RUN_ALL_TESTS();
}