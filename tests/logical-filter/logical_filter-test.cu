#include <cstdlib>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include <LogicalFilter.h>
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

struct logical_filter_TEST : public ::testing::Test {

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
		inputs[0] = create_gdf_column(GDF_INT8, num_values, (void *) input1, 1);
		inputs[1] = create_gdf_column(GDF_INT8, num_values, (void *) input2, 1);
		inputs[2] = create_gdf_column(GDF_INT8, num_values, (void *) input3, 1);

		raw_inputs.resize(3);
		raw_inputs[0] = inputs[0].get();
		raw_inputs[1] = inputs[1].get();
		raw_inputs[2] = inputs[2].get();

		blzframe.add_table(raw_inputs);
	
		output = create_gdf_column(GDF_INT8, num_values, nullptr, 1);
	
		temp = create_gdf_column(GDF_INT8, num_values, nullptr, 1);
	
		host_output = new char[num_values];
		device_output = new char[num_values];
	}

	void TearDown(){

		cudaMemcpy(device_output, output->data, num_values * WIDTH_PER_VALUE, cudaMemcpyDeviceToHost);

		for(int i = 0; i < num_values; i++){
			EXPECT_TRUE(host_output[i] == device_output[i]);
		}

		/*for(int i = 0; i < outputs.size(); i++){
			print_column(outputs[i]);
		}*/

		delete[] input1;
		delete[] input2;
		delete[] input3;

		delete[] host_output;
		delete[] device_output;
	}

	std::vector<gdf_col_pointer> inputs;
	std::vector<gdf_column *> raw_inputs;

	size_t num_values = 32;

	void * temp_space;

	char * input1;
	char * input2;
	char * input3;

	blazing_frame blzframe;

	gdf_col_pointer output, temp;

	char * host_output;
	char * device_output;

	const int WIDTH_PER_VALUE = 1;
};

TEST_F(logical_filter_TEST, processing_expressions0) {

	{
		std::string expression = ">($1, 5)";

		evaluate_expression(
				blzframe,
				expression,
				output.get(),
				temp.get());

		for(int i = 0; i < num_values; i++){
			host_output[i] = input2[i] > 5 ? 1 : 0;
		}
	}
}

TEST_F(logical_filter_TEST, processing_expressions1) {

	{
		std::string expression = "+(*($1, $0), $2)";

		evaluate_expression(
				blzframe,
				expression,
				output.get(),
				temp.get());

		for(int i = 0; i < num_values; i++){
			host_output[i] = ((input1[i] * input2[i]) + input3[i]);
		}
	}
}

TEST_F(logical_filter_TEST, processing_expressions2) {

	{
		std::string expression = "=(=($1, $0), $2)";

		evaluate_expression(
				blzframe,
				expression,
				output.get(),
				temp.get());

		for(int i = 0; i < num_values; i++){
			host_output[i] = ((input1[i] == input2[i]) == input3[i]) ? 1 : 0;
		}
    }
}

TEST_F(logical_filter_TEST, processing_expressions3) {

	{
		std::string expression = "*($0, $0)";

		evaluate_expression(
				blzframe,
				expression,
				output.get(),
				temp.get());

		for(int i = 0; i < num_values; i++){
			host_output[i] = input1[i] * input1[i];
		}
    }
}

TEST_F(logical_filter_TEST, processing_expressions4) {

	{
		std::string expression = "+($0, $1)";

		evaluate_expression(
				blzframe,
				expression,
				output.get(),
				temp.get());

		for(int i = 0; i < num_values; i++){
			host_output[i] = input1[i] + input2[i];
		}
	}
}

TEST_F(logical_filter_TEST, processing_expressions5) {

	{
		std::string expression = "=(*($0, $0), 1))";

		evaluate_expression(
				blzframe,
				expression,
				output.get(),
				temp.get());

		for(int i = 0; i < num_values; i++){
			host_output[i] = ((input1[i] * input1[i]) == 1) ? 1 : 0;
		}
    }
}

/*TEST_F(logical_filter_TEST, processing_expressions6) {

	{
		std::string expression = "AND($0, $1)";

		evaluate_expression(
				blzframe,
				expression,
				output,
				temp);

		for(int i = 0; i < num_values; i++){
			host_output[i] = (input1[i] && input2[i]) ? 1 : 0;
		}
    }
}

TEST_F(logical_filter_TEST, processing_expressions7) {

	{
		std::string expression = "AND(=(*($0, $0), 1), =($1, 2))";

		evaluate_expression(
				blzframe,
				expression,
				output,
				temp);

		for(int i = 0; i < num_values; i++){
			host_output[i] = ((input1[i] * input1[i]) == 1) && (input2[i] == 2) ? 1 : 0;
		}
	}
}*/

int main(int argc, char **argv){
	::testing::InitGoogleTest(&argc, argv);
	::testing::Environment* const env = ::testing::AddGlobalTestEnvironment(new TestEnvironment());
	return RUN_ALL_TESTS();
}