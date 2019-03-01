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
//#include <Utils.cuh>

#include <blazingdb/io/Library/Logging/Logger.h>
#include <blazingdb/io/Library/Logging/CoutOutput.h>
#include "blazingdb/io/Library/Logging/ServiceLogging.h"

#include <NVCategory.h>
#include <utilities/bit_mask.h>

class TestEnvironment : public testing::Environment {
public:
	virtual ~TestEnvironment() {}
	virtual void SetUp() {
		auto output = new Library::Logging::CoutOutput();
  		Library::Logging::ServiceLogging::getInstance().setLogOutput(output);
	}

	void TearDown() {
		cudaDeviceReset(); //for cuda-memchecking
	}
};

namespace {
std::string const default_chars = 
	//"abcdefghijklmnaoqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
	//"abcdefghijklmnaoqrstuvwxyz";
	"hijklmnaoq";
}

struct NVCategoryTest : public ::testing::Test {
	
	std::string random_string(size_t len = 15, std::string const &allowed_chars = default_chars) {
		std::mt19937_64 gen { std::random_device()() };
		std::uniform_int_distribution<size_t> dist { 0, allowed_chars.length()-1 };
	
		std::string ret;
		std::generate_n(std::back_inserter(ret), len, [&] { return allowed_chars[dist(gen)]; });
		return ret;
	}

	gdf_column * create_boolean_column(gdf_size_type num_rows){
		gdf_column * column = new gdf_column;
		int * data;
		bit_mask::bit_mask_t * valid;
		bit_mask::create_bit_mask(&valid, num_rows,1);
		EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(int8_t) , 0), RMM_SUCCESS);
		gdf_error err = gdf_column_view(column,
				(void *) data,
				(gdf_valid_type *) valid,
				num_rows,
				GDF_INT8);
		return column;

	}


	gdf_column * create_column_constant(gdf_size_type num_rows, int value){
		gdf_column * column = new gdf_column;
		int * data;
		bit_mask::bit_mask_t * valid;
		bit_mask::create_bit_mask(&valid, num_rows,1);
		EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(int) , 0), RMM_SUCCESS);
		cudaMemset(data,value,sizeof(int) * num_rows);
		gdf_error err = gdf_column_view(column,
				(void *) data,
				(gdf_valid_type *) valid,
				num_rows,
				GDF_INT32);
		return column;

	}


	gdf_column * create_indices_column(gdf_size_type num_rows){
		gdf_column * column = new gdf_column;
		int * data;
		bit_mask::bit_mask_t * valid;
		bit_mask::create_bit_mask(&valid, num_rows,1);
		EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(int) , 0), RMM_SUCCESS);
		gdf_error err = gdf_column_view(column,
				(void *) data,
				(gdf_valid_type *) valid,
				num_rows,
				GDF_INT32);
		return column;

	}

	gdf_column * create_column_ints(int32_t* host_data, gdf_size_type num_rows){
		gdf_column * column = new gdf_column;
		int32_t * data;
		EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(int32_t) , 0), RMM_SUCCESS);
		cudaMemcpy(data, host_data, sizeof(int32_t) * num_rows, cudaMemcpyHostToDevice);
	
		bit_mask::bit_mask_t * valid;
		bit_mask::create_bit_mask(&valid, num_rows,1);
	
		gdf_error err = gdf_column_view(column,
				(void *) data,
				(gdf_valid_type *)valid,
				num_rows,
				GDF_INT32);
		return column;
	}

	gdf_column * create_nv_category_column(gdf_size_type num_rows, bool repeat_strings){

		const char ** string_host_data = new const char *[num_rows];


		for(gdf_size_type row_index = 0; row_index < num_rows; row_index++){
			string_host_data[row_index] = new char[(num_rows + 25) / 26]; //allows string to grow depending on numbe of rows
			std::string temp_string = "";
			int num_chars = repeat_strings ? 1 : (row_index / 26) + 1;
			char repeat_char = (26 - (row_index % 26)) + 65; //chars are Z,Y ...C,B,A,ZZ,YY,.....BBB,AAA.....
			for(int char_index = 0; char_index < num_chars; char_index++){
				temp_string.push_back(repeat_char);
			}
			temp_string.push_back(0);
			std::memcpy((void *) string_host_data[row_index],temp_string.c_str(),temp_string.size());

		}

		NVCategory* category = NVCategory::create_from_array(string_host_data, num_rows);

		gdf_column * column = new gdf_column;
		int * data;
		EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(gdf_nvstring_category) , 0), RMM_SUCCESS);


		category->get_values( (int *)data, true );
		bit_mask::bit_mask_t * valid;
		bit_mask::create_bit_mask(&valid, num_rows,1);

		gdf_error err = gdf_column_view(column,
				(void *) data,
				(gdf_valid_type *)valid,
				num_rows,
				GDF_STRING_CATEGORY);
		column->dtype_info.category = category;
		return column;
	}

	gdf_column * create_nv_category_column_from_strings(std::vector<std::string>& strings){

		size_t num_rows = strings.size();
		const char ** string_host_data = new const char *[num_rows];

		for(gdf_size_type row_index = 0; row_index < num_rows; row_index++){
			string_host_data[row_index] = new char[200];
			strings[row_index].push_back(0);
			std::memcpy((void *) string_host_data[row_index],strings[row_index].c_str(),strings[row_index].size());
		}

		NVCategory* category = NVCategory::create_from_array(string_host_data, num_rows);

		gdf_column * column = new gdf_column;
		int * data;
		EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(gdf_nvstring_category) , 0), RMM_SUCCESS);

		category->get_values( (int *)data, true );
		bit_mask::bit_mask_t * valid;
		bit_mask::create_bit_mask(&valid, num_rows,1);

		gdf_error err = gdf_column_view(column,
				(void *) data,
				(gdf_valid_type *)valid,
				num_rows,
				GDF_STRING_CATEGORY);
		column->dtype_info.category = category;
		return column;
	}

	int32_t* generate_int_data(gdf_size_type num_rows, size_t max_value, bool print=false){
		int32_t* host_data = new int32_t[num_rows];
	
		for(gdf_size_type row_index = 0; row_index < num_rows; row_index++){
			host_data[row_index] = std::rand() % max_value;
	
			if(print)
				std::cout<<host_data[row_index]<<"\t";
		}
		if(print)
			std::cout<<std::endl;
	
		return host_data;
	}

	gdf_column * create_nv_category_column_strings(const char ** string_host_data, gdf_size_type num_rows){
		NVCategory* category = NVCategory::create_from_array(string_host_data, num_rows);
	
		gdf_column * column = new gdf_column;
		int * data;
		EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(gdf_nvstring_category) , 0), RMM_SUCCESS);
	
		category->get_values( (int *)data, true );
		bit_mask::bit_mask_t * valid;
		bit_mask::create_bit_mask(&valid, num_rows,1);
	
		gdf_error err = gdf_column_view(column,
				(void *) data,
				(gdf_valid_type *)valid,
				num_rows,
				GDF_STRING_CATEGORY);
		column->dtype_info.category = category;
		return column;
	}

	const char ** generate_string_data(gdf_size_type num_rows, size_t length, bool print=false){
		const char ** string_host_data = new const char *[num_rows];
	
		for(gdf_size_type row_index = 0; row_index < num_rows; row_index++){
			string_host_data[row_index] = new char[length+1];
	
			std::string rand_string = random_string(length);
			rand_string.push_back(0);
			if(print)
				std::cout<<rand_string<<"\t";
			std::memcpy((void *) string_host_data[row_index],rand_string.c_str(),rand_string.size());
		}
		if(print)
			std::cout<<std::endl;
	
		return string_host_data;
	}

	void SetUp(){

		/*input1 = new char[num_values];
		input2 = new char[num_values];

		for(int i = 0; i < num_values; i++){
			if(i % 2 == 0){
				input1[i] = 1;
			}else{
				input1[i] = i;
			}
			input2[i] = i;
		}

		gdf_column * column = create_nv_category_column(num_values,false);

		inputs.resize(2);
		inputs[0].create_gdf_column(GDF_INT8, num_values, (void *) input1, 1);
		inputs[1].create_gdf_column(column);

		input_tables.push_back(inputs); //columns for emps
		input_tables.push_back(inputs); //columns for sales*/
	}

	void TearDown(){

		for(int i = 0; i < outputs.size(); i++){
			print_column<int8_t>(outputs[i].get_gdf_column());

			// Releasing allocated memory, here we are responsible for that
			//TODO percy rommel: move to integration/end-to-end test
			//GDFRefCounter::getInstance()->free_if_deregistered(outputs[i].get_gdf_column());
		}
	}

	void Check(gdf_column_cpp out_col, int32_t* host_output, size_t num_output_values){

		if(num_output_values == 0)
			num_output_values = out_col.size();

		int32_t * device_output;
		device_output = new int32_t[num_output_values];
		cudaMemcpy(device_output, out_col.data(), num_output_values * sizeof(int32_t), cudaMemcpyDeviceToHost);

		for(int i = 0; i < num_output_values; i++){
			EXPECT_TRUE(host_output[i] == device_output[i]);
		}
	}

	void Check(gdf_column_cpp out_col, std::vector<std::string> reference_result){

		const size_t num_values = out_col.size();
		NVStrings * temp_strings = out_col.get_gdf_column()->dtype_info.category->to_strings();

		char** host_strings = new char*[num_values];
		temp_strings->to_host(host_strings, 0, num_values);

		std::cout<<"Converted:\n";
		for(size_t I=0;I<num_values;I++){
			std::cout<<std::string(host_strings[I])<<std::endl;
		}
		std::cout<<"\n";

		EXPECT_EQ(out_col.size(), reference_result.size()) << "Mismatch columns size";
		
		for(int i = 0; i < reference_result.size(); i++){
			EXPECT_TRUE(reference_result[i] == std::string(host_strings[i]));
		}

		NVStrings::destroy(temp_strings);
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

	std::vector<gdf_column_cpp> inputs;

	char * input1;
	char * input2;

	size_t num_values = 64;

	std::vector<std::vector<gdf_column_cpp> > input_tables;
	std::vector<std::string> table_names={"hr.emps", "hr.sales"};
	std::vector<std::vector<std::string>> column_names={{"x", "y"},{"a", "b"}};

	std::vector<gdf_column_cpp> outputs;

	const int WIDTH_PER_VALUE = 1;
};

TEST_F(NVCategoryTest, DISABLED_processing_filter_wo_strings) {

	{   //select x,y from hr.emps where y='BBB'
		std::string query = "LogicalProject(x=[$0])\n\
	EnumerableTableScan(table=[[hr, emps]])";

		std::cout<<"about to evalute"<<std::endl;
		gdf_error err = evaluate_query(input_tables, table_names, column_names,
				query, outputs);
		std::cout<<"evaluated"<<std::endl;
		EXPECT_TRUE(err == GDF_SUCCESS);
	}
}

TEST_F(NVCategoryTest, processing_filter_comparison_right_string) {

	{   //select x from hr.emps where y<'Z'

		bool print = true;
		size_t length = 1;

		int32_t* host_data = generate_int_data(num_values, 10, print);
		const char ** string_data = generate_string_data(num_values, length, print);

		gdf_column * string_column = create_nv_category_column_strings(string_data, num_values);

		inputs.resize(2);
		inputs[0].create_gdf_column(GDF_INT32, num_values, (void *) host_data, sizeof(int32_t));
		inputs[1].create_gdf_column(string_column);

		input_tables.push_back(inputs);
		input_tables.push_back(inputs);

		std::string query = "LogicalProject(x=[$0])\n\
	LogicalFilter(condition=[<($1, 'm')])\n\
		EnumerableTableScan(table=[[hr, emps]])";

		std::cout<<"about to evalute"<<std::endl;
		gdf_error err = evaluate_query(input_tables, table_names, column_names,
				query, outputs);
		EXPECT_TRUE(err == GDF_SUCCESS);
		std::cout<<"evaluated"<<std::endl;

		std::vector<int32_t> reference_result;
		for(size_t I=0; I<num_values; I++){
			if(std::string(string_data[I]) < "m"){
				reference_result.push_back(host_data[I]);
				std::cout<<host_data[I]<<" ";
			}
		}
		std::cout<<std::endl;

		print_gdf_column(outputs[0].get_gdf_column());
		Check(outputs[0], reference_result.data(), reference_result.size());
	}
}

TEST_F(NVCategoryTest, processing_filter_comparison_both_strings) {

	{   //select * from hr.emps where x=y

		bool print = true;
		size_t length = 2;

		const char ** left_string_data = generate_string_data(num_values, length, print);
		const char ** right_string_data = generate_string_data(num_values, length, print);

		gdf_column * left_string_column = create_nv_category_column_strings(left_string_data, num_values);
		gdf_column * right_string_column = create_nv_category_column_strings(right_string_data, num_values);

		std::cout<<"Input:\n";
		print_gdf_column(left_string_column);
		print_gdf_column(right_string_column);

		inputs.resize(2);
		inputs[0].create_gdf_column(left_string_column);
		inputs[1].create_gdf_column(right_string_column);

		input_tables.push_back(inputs);
		input_tables.push_back(inputs);

		std::string query = "LogicalProject(x=[$0], y=[$1])\n\
	LogicalFilter(condition=[=($1, $0)])\n\
		EnumerableTableScan(table=[[hr, emps]])";

		std::cout<<"about to evalute"<<std::endl;
		gdf_error err = evaluate_query(input_tables, table_names, column_names,
				query, outputs);
		EXPECT_TRUE(err == GDF_SUCCESS);
		std::cout<<"evaluated"<<std::endl;

		std::vector<std::string> left_reference_result;
		std::vector<std::string> right_reference_result;
		for(size_t I=0; I<num_values; I++){
			if(std::string(left_string_data[I]) == std::string(right_string_data[I])){
				left_reference_result.push_back(left_string_data[I]);
				right_reference_result.push_back(right_string_data[I]);
				std::cout<<std::string(left_string_data[I])<<"- : -"<<std::string(right_string_data[I])<<"\n";
			}
		}
		std::cout<<std::endl;

		std::cout<<"Output:\n";
		print_gdf_column(outputs[0].get_gdf_column());
		print_gdf_column(outputs[1].get_gdf_column());

		Check(outputs[0], left_reference_result);
		Check(outputs[1], right_reference_result);
	}
}

int main(int argc, char **argv){
	::testing::InitGoogleTest(&argc, argv);
	::testing::Environment* const env = ::testing::AddGlobalTestEnvironment(new TestEnvironment());
	return RUN_ALL_TESTS();
}
