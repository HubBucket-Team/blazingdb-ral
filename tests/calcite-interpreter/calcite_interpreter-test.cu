#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

#include "gtest/gtest.h"
#include <CalciteExpressionParsing.h>
#include <CalciteInterpreter.h>
#include <blazingdb/io/Util/StringUtil.h>
#include <DataFrame.h>
#include <GDFColumn.cuh>
#include <GDFCounter.cuh>
#include <Utils.cuh>

//#include <sqls_rtti_comp.hpp> //TODO build fails here it seems we need to export this header from libgdf

class TestEnvironment : public testing::Environment {
public:
	virtual ~TestEnvironment() {}
	virtual void SetUp() {}

	void TearDown() {
		cudaDeviceReset(); //for cuda-memchecking
	}
};

struct calcite_interpreter_TEST : public ::testing::Test {

	void SetUp(){

		input1 = new char[num_values];
		input2 = new char[num_values];
		input3 = new char[num_values];

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

		/*print_column(inputs[0].get_gdf_column());
		print_column(inputs[1].get_gdf_column());
		print_column(inputs[2].get_gdf_column());*/

		input_tables.push_back(inputs); //columns for emps
		input_tables.push_back(inputs); //columns for sales
	}

	void TearDown(){

		for(int i = 0; i < outputs.size(); i++){
			print_column<int8_t>(outputs[i].get_gdf_column());

			// Releasing allocated memory, here we are responsible for that
			//TODO percy rommel: move to integration/end-to-end test
			//GDFRefCounter::getInstance()->free_if_deregistered(outputs[i].get_gdf_column());
		}
	}

	void Check(gdf_column_cpp out_col, char* host_output){

		char * device_output;
		device_output = new char[out_col.size()];
		cudaMemcpy(device_output, out_col.data(), out_col.size() * WIDTH_PER_VALUE, cudaMemcpyDeviceToHost);

		for(int i = 0; i < out_col.size(); i++){
			//std::cout<<(int)host_output[i]<<" =?= "<<(int)device_output[i]<<std::endl<<std::flush;
			EXPECT_TRUE(host_output[i] == device_output[i]);
		}
	}

	gdf_column_cpp left;
	gdf_column_cpp right;
	gdf_column_cpp third;

	std::vector<gdf_column_cpp> inputs;

	char * input1;
	char * input2;
	char * input3;

	size_t num_values = 32;

	std::vector<std::vector<gdf_column_cpp> > input_tables;
	std::vector<std::string> table_names={"hr.emps", "hr.sales"};
	std::vector<std::vector<std::string>> column_names={{"x", "y", "z"},{"a", "b", "x"}};

	std::vector<gdf_column_cpp> outputs;

	const int WIDTH_PER_VALUE = 1;
};

/*TEST_F(calcite_interpreter_TEST, processing_project0) {

	{   //select * from hr.emps
		std::string query = "\
LogicalProject(x=[$0], y=[$1], z=[$2])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 3);

		Check(outputs[0], input1);
		Check(outputs[1], input2);
		Check(outputs[2], input3);
	}
}

TEST_F(calcite_interpreter_TEST, processing_project1) {

	{   //select x from hr.emps
		std::string query = "\
LogicalProject(x=[$0])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);

		Check(outputs[0], input1);
	}
}

TEST_F(calcite_interpreter_TEST, processing_project2) {

	{ //select z > 5 from hr.emps
		std::string query = "\
LogicalProject(EXPR$0=[>($2, 5)])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);

		char * host_output = new char[num_values];
		for(int i = 0; i < num_values; i++){
			host_output[i] = input3[i] > 5 ? 1 : 0;
		}

		Check(outputs[0], host_output);
	}
}

TEST_F(calcite_interpreter_TEST, processing_project3) {

	{   //select a from hr.sales
		std::string query = "\
LogicalProject(a=[$0])\n\
  EnumerableTableScan(table=[[hr, sales]])";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);

		Check(outputs[0], input1);
	}
}

TEST_F(calcite_interpreter_TEST, processing_project4) {

	{   //select x + y, z from hr.emps
		std::string query = "\
LogicalProject(EXPR$0=[+($0, $1)], z=[$2])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 2);

		char * host_output = new char[num_values];
		for(int i = 0; i < num_values; i++){
			host_output[i] = input1[i] + input2[i];
		}

		Check(outputs[0], host_output);
		Check(outputs[1], input3);
	}
}

/*TEST_F(calcite_interpreter_TEST, processing_project5) {

	{   //select z from hr.emps where x = y
		std::string query = "\
LogicalProject(z=[$2])\n\
  LogicalFilter(condition=[=($0, $1)])\n\
    EnumerableTableScan(table=[[hr, emps]])";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
			query, outputs);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);

		int cur = 0;
		char * host_output = new char[num_values];
		for(int i = 0; i < num_values; i++){
			if(input1[i] == input2[i]){
				host_output[cur] = input3[i];
				cur++;
			}
		}

		Check(outputs[0], host_output);
	}
}*/

TEST_F(calcite_interpreter_TEST, processing_project6) {

	{   //select x - y as S from hr.emps
		std::string query = "\
LogicalProject(S=[-($0, $1)])\n\
  EnumerableTableScan(table=[[hr, emps]])";

		gdf_error err = evaluate_query(input_tables, table_names, column_names,
				query, outputs);
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 1);

		char * host_output = new char[num_values];
		for(int i = 0; i < num_values; i++){
			host_output[i] = input1[i] - input2[i];
		}

		Check(outputs[0], host_output);
	}
}

TEST_F(calcite_interpreter_TEST, order_by) {

	{
		size_t num_values = 32;

		uint64_t * data_test = new uint64_t[num_values];
		int* data = new int[num_values];
		for(size_t i = 0; i < num_values; i++){
			data[i] = num_values - i;
			data_test[i] = i;
		}

		gdf_column_cpp input_column;

		gdf_column_cpp indices_col;

		 std::cout<<"now running other version"<<std::endl;


		input_column.create_gdf_column(GDF_INT32,num_values,(void *) data,4);


		//TODO percy noboa see upgrade to uints
		//indices_col.create_gdf_column(GDF_UINT64,num_values,nullptr,8);
		indices_col.create_gdf_column(GDF_INT64,num_values,nullptr,8);
		print_column<uint64_t>(indices_col.get_gdf_column());
		gdf_valid_type asc_desc_bitmask = 255;
		gdf_valid_type* asc_desc_bitmask_dev;

		cudaMalloc((void **) &asc_desc_bitmask_dev,1);

		cudaError_t err2 = cudaMemcpy(asc_desc_bitmask_dev,&asc_desc_bitmask,1,cudaMemcpyHostToDevice);


		std::vector<gdf_column> v_cols(1);
		for(auto i = 0; i < 1; ++i)
		{
			v_cols[i] = *(input_column.get_gdf_column());
		}

		gdf_column* input_columns = &v_cols[0];


		print_column<int32_t>(input_column.get_gdf_column());
		try{
		    //TODO percy noboa felipe see upgrade to order_by
//		    gdf_error err = gdf_order_by_asc_desc(
//					input_columns,
//					1,
//					indices_col.get_gdf_column(),
//					asc_desc_bitmask_dev);
//			EXPECT_TRUE(err == GDF_SUCCESS);

		}catch(std::exception e){

			std::cout<<"We caught an exception running order by!"<<e.what()<<std::endl;
		}
		std::cout<<"printing size "<<indices_col.size()<<std::endl;
		print_typed_column<uint64_t>((uint64_t *) indices_col.get_gdf_column()->data, nullptr,indices_col.size());
		delete[] data;
		cudaFree(asc_desc_bitmask_dev);

		// thrust::device_vector<void*> d_cols(1, nullptr);
		// thrust::device_vector<int>   d_types(1, 0);


		/*gdf_column col;


		void * col_data;
		cudaMalloc(&col_data,num_values * sizeof(int32_t));
		gdf_error err_create = gdf_column_view(&col, col_data, nullptr, num_values, GDF_INT32);
		void ** d_cols;
		cudaMalloc( &d_cols,sizeof(void*) * 1);
		int * d_types;
		cudaMalloc(&d_types,sizeof(int) * 1);
		gdf_column * cols = new gdf_column[1];
		cols[0] = col;
		std::cout<<"yes this one!"<<std::endl;
		size_t * indices;
		cudaMalloc(&indices,sizeof(size_t) * num_values);
		gdf_error err = gdf_order_by(num_values,
				cols,
				1,
				d_cols,
				d_types,
				indices);
		//				(size_t *) indices.data());

		if(err != GDF_SUCCESS){
			std::cout<<"We had an issue!!!"<<std::endl;
		}
		cudaFree(d_cols);
		cudaFree(col_data);
		cudaFree(d_types);
		//		gdf_column indices_col;
		//		gdf_column_view(&indices_col, indices, nullptr, num_values, GDF_UINT64);

		//print_column(indices_col);
		EXPECT_TRUE(err == GDF_SUCCESS);*/

	}
}

TEST_F(calcite_interpreter_TEST, processing_sort) {

	{   //select x - y as S from hr.emps
		std::string query = "LogicalSort(sort0=[$0], dir0=[ASC])\n\
  LogicalProject(x=[$0], x=[$1])\n\
    EnumerableTableScan(table=[[hr, emps]])";
		std::cout<<"about to evalute"<<std::endl;
		gdf_error err = evaluate_query(input_tables, table_names, column_names,
				query, outputs);
		std::cout<<"evaluated"<<std::endl;
		EXPECT_TRUE(err == GDF_SUCCESS);
		EXPECT_TRUE(outputs.size() == 2);

		for(int i = 0; i < outputs.size(); i++){
			print_column<int8_t>(outputs[i].get_gdf_column());
		}

		std::vector<char> output = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29,31};
		char * host_output = &output[0];
		Check(outputs[0], host_output);
		output = {0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
		host_output = &output[0];
		Check(outputs[1], host_output);
	}
}


int main(int argc, char **argv){
	::testing::InitGoogleTest(&argc, argv);
	::testing::Environment* const env = ::testing::AddGlobalTestEnvironment(new TestEnvironment());
	return RUN_ALL_TESTS();
}
