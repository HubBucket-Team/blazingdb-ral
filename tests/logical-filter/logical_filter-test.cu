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
		/*raw_inputs.push_back(&left);
		raw_inputs.push_back(&right);
		raw_inputs.push_back(&third);*/

		raw_inputs.resize(3);
		/*raw_inputs[0] = &left;
		raw_inputs[1] = &right;
		raw_inputs[2] = &third;*/

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

		/*raw_inputs.resize(3);
		create_gdf_column(raw_inputs[0], GDF_INT8, num_values, (void *) input1, 1);
		create_gdf_column(raw_inputs[1], GDF_INT8, num_values, (void *) input2, 1);
		create_gdf_column(raw_inputs[2], GDF_INT8, num_values, (void *) input3, 1);*/

		inputs.resize(3);
		inputs[0] = create_gdf_column(GDF_INT8, num_values, (void *) input1, 1);
		inputs[1] = create_gdf_column(GDF_INT8, num_values, (void *) input2, 1);
		inputs[2] = create_gdf_column(GDF_INT8, num_values, (void *) input3, 1);

		raw_inputs[0] = inputs[0].get();
		raw_inputs[1] = inputs[1].get();
		raw_inputs[2] = inputs[2].get();

		blzframe.add_table(raw_inputs);
	
		//output = new gdf_column;
		//create_gdf_column(output, GDF_INT8, num_values, nullptr, 1);
		output = create_gdf_column(GDF_INT8, num_values, nullptr, 1);
	
		//temp = new gdf_column;
		//create_gdf_column(temp, GDF_INT8, num_values, nullptr, 1);
		temp = create_gdf_column(GDF_INT8, num_values, nullptr, 1);
	
		host_output = new char[num_values];
		device_output = new char[num_values];
	}

	void TearDown(){
		/*for(int i = 0; i < outputs.size(); i++){
			print_column(outputs[i]);
		}*/

		delete[] input1;
		delete[] input2;
		delete[] input3;
	}

	/*gdf_column left;
	gdf_column right;
	gdf_column third;*/

	std::vector<gdf_col_pointer> inputs;
	std::vector<gdf_column *> raw_inputs;

	size_t num_values = 32;

	void * temp_space;

	char * input1;
	char * input2;
	char * input3;

	blazing_frame blzframe;

	gdf_col_pointer output, temp;

	/*gdf_column * output;
	gdf_column * temp;*/

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
	
		cudaMemcpy(device_output, output->data, num_values * WIDTH_PER_VALUE, cudaMemcpyDeviceToHost);

		for(int i = 0; i < num_values; i++){
			EXPECT_TRUE(host_output[i] == device_output[i]);
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
	
		cudaMemcpy(device_output, output->data, num_values * WIDTH_PER_VALUE, cudaMemcpyDeviceToHost);

		for(int i = 0; i < num_values; i++){
			EXPECT_TRUE(host_output[i] == device_output[i]);
		}
	}
}

/*
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
		std::string expression = "+($0, $1)";

		evaluate_expression(
				blzframe,
				expression,
				output,
				temp);

		for(int i = 0; i < num_values; i++){
			host_output[i] = input1[i] + input2[i];
		}

		cudaMemcpy(device_output, output->data, num_values * WIDTH_PER_VALUE, cudaMemcpyDeviceToHost);
		
		//std::cout<<"Left\n";
		//print_column(&left);
		//std::cout<<"EndLeft\n";

		//std::cout<<"Right\n";
		//print_column(&right);
		//std::cout<<"EndRight\n";

		//std::cout<<"Output\n";
		for(int i = 0; i < num_values; i++){
			//std::cout<<(int)host_output[i]<<"\n";
			EXPECT_TRUE(host_output[i] == device_output[i]);
		}
		//std::cout<<"EndOutput\n";
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

int main(int argc, char **argv){
	::testing::InitGoogleTest(&argc, argv);
	::testing::Environment* const env = ::testing::AddGlobalTestEnvironment(new TestEnvironment());
	return RUN_ALL_TESTS();
}