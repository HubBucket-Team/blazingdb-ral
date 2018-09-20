#include <cstdlib>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include <LogicalFilter.h>
#include <DataFrame.h>
#include <Utils.cuh>
#include <GDFColumn.cuh>

#include <gdf/gdf.h>

class TestEnvironment : public testing::Environment {
public:
	virtual ~TestEnvironment() {}
	virtual void SetUp() {}

	void TearDown() {
		cudaDeviceReset();
	}
};

TEST(logical_filter_TEST, processing_expressions) {

	// Initializating data
	std::vector<gdf_column_cpp> inputs_(3);

	/*gdf_column left;
	gdf_column right;
	gdf_column third;

	std::vector<gdf_column *> inputs(3);
	inputs[0] = &left;
	inputs[1] = &right;
	inputs[2] = &third;*/

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

	inputs_[0].create_gdf_column(GDF_INT8, num_values, (void *) input1, 1);
	inputs_[1].create_gdf_column(GDF_INT8, num_values, (void *) input2, 1);
	inputs_[2].create_gdf_column(GDF_INT8, num_values, (void *) input3, 1);

	/*create_gdf_column(inputs[0], GDF_INT8, num_values, (void *) input1, 1);
	create_gdf_column(inputs[1], GDF_INT8, num_values, (void *) input2, 1);
	create_gdf_column(inputs[2], GDF_INT8, num_values, (void *) input3, 1);*/

	/*blazing_frame blzframe;
	blzframe.add_table(inputs);

	gdf_column * output = new gdf_column;
	create_gdf_column(output, GDF_INT8, num_values, nullptr, 1);

	gdf_column * temp = new gdf_column;
	create_gdf_column(temp, GDF_INT8, num_values, nullptr, 1);

	char * host_output = new char[num_values];
	char * device_output = new char[num_values];

	const int WIDTH_PER_VALUE = 1;*/
    /*{
		std::string expression = ">($1, 5)";

		evaluate_expression(
				blzframe,
				expression,
				output,
				temp);

		for(int i = 0; i < num_values; i++){
			host_output[i] = input2[i] > 5 ? 1 : 0;
		}
	
		cudaMemcpy(device_output, output->data, num_values * WIDTH_PER_VALUE, cudaMemcpyDeviceToHost);

		for(int i = 0; i < num_values; i++){
			EXPECT_TRUE(host_output[i] == device_output[i]);
		}
	}

	{
		std::string expression = "+(*($1, $0), $2)";

		evaluate_expression(
				blzframe,
				expression,
				output,
				temp);

		for(int i = 0; i < num_values; i++){
			host_output[i] = ((input1[i] * input2[i]) + input3[i]);
		}
	
		cudaMemcpy(device_output, output->data, num_values * WIDTH_PER_VALUE, cudaMemcpyDeviceToHost);

		for(int i = 0; i < num_values; i++){
			EXPECT_TRUE(host_output[i] == device_output[i]);
		}
	}

	{
		std::string expression = "=(=($1, $0), $2)";

		evaluate_expression(
				blzframe,
				expression,
				output,
				temp);

		for(int i = 0; i < num_values; i++){
			host_output[i] = ((input1[i] == input2[i]) == input3[i]) ? 1 : 0;
		}
	
		cudaMemcpy(device_output, output->data, num_values * WIDTH_PER_VALUE, cudaMemcpyDeviceToHost);

		for(int i = 0; i < num_values; i++){
			EXPECT_TRUE(host_output[i] == device_output[i]);
		}
    }

	{
		std::string expression = "*($0, $0)";

		evaluate_expression(
				blzframe,
				expression,
				output,
				temp);

		for(int i = 0; i < num_values; i++){
			host_output[i] = input1[i] * input1[i];
		}

		cudaMemcpy(device_output, output->data, num_values * WIDTH_PER_VALUE, cudaMemcpyDeviceToHost);

		for(int i = 0; i < num_values; i++){
			EXPECT_TRUE(host_output[i] == device_output[i]);
		}
    }

	{
		std::string expression = "=(*($0, $0), 1))";

		evaluate_expression(
				blzframe,
				expression,
				output,
				temp);

		for(int i = 0; i < num_values; i++){
			host_output[i] = ((input1[i] * input1[i]) == 1) ? 1 : 0;
		}
	
		cudaMemcpy(device_output, output->data, num_values * WIDTH_PER_VALUE, cudaMemcpyDeviceToHost);

		for(int i = 0; i < num_values; i++){
			EXPECT_TRUE(host_output[i] == device_output[i]);
		}
    }*/

	/*{
		std::string expression = "AND($0, $1)";

		evaluate_expression(
				blzframe,
				expression,
				output,
				temp);

		for(int i = 0; i < num_values; i++){
			host_output[i] = (input1[i] && input2[i]) ? 1 : 0;
		}
	
		cudaMemcpy(device_output, output->data, num_values * WIDTH_PER_VALUE, cudaMemcpyDeviceToHost);

		for(int i = 0; i < num_values; i++){
			EXPECT_TRUE(host_output[i] == device_output[i]);
		}
    }*/

	/*{
		std::string expression = "AND(=(*($0, $0), 1), =($1, 2))";

		evaluate_expression(
				blzframe,
				expression,
				output,
				temp);

		for(int i = 0; i < num_values; i++){
			host_output[i] = ((input1[i] * input1[i]) == 1) && (input2[i] == 2) ? 1 : 0;
		}
	
		cudaMemcpy(device_output, output->data, num_values * WIDTH_PER_VALUE, cudaMemcpyDeviceToHost);

		for(int i = 0; i < num_values; i++){
			EXPECT_TRUE(host_output[i] == device_output[i]);
		}
    }*/
}

int main(int argc, char **argv){
	::testing::InitGoogleTest(&argc, argv);
	::testing::Environment* const env = ::testing::AddGlobalTestEnvironment(new TestEnvironment());
	return RUN_ALL_TESTS();
}