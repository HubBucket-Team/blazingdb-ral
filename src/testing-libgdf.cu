/*
 ============================================================================
 Name        : testing-libgdf.cu
 Author      : felipe
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include <cuda_runtime.h>
#include "LogicalFilter.h"
#include "CalciteInterpreter.h"
#include "DataFrame.h"
#include "Utils.cuh"
#include <cuda.h>

#include "ipc/calcite_client.h"
#include "gdf/gdf.h"

#include <blazingdb/protocol/api.h>

#define BIT_FIVE 0x10
#define BIT_SIX 0x20

struct DMLRequest
{
  char query[1024]; //logicalPlan
};

struct DMLResponse
{
  uint64_t query_token; //token
};

/*void runOriginalTest(){
	gdf_size_type num_elements = 8;

		char * data_left;
		char * data_right;
		char * data_out;
		cudaError_t cuda_error = cudaMalloc((void **) &data_left,sizeof(int8_t) * num_elements);
		cuda_error = cudaMalloc((void **) &data_right,sizeof(int8_t) * num_elements);
		cuda_error = cudaMalloc((void **) &data_out,sizeof(int8_t) * num_elements);

		thrust::device_ptr<int8_t> left_ptr= thrust::device_pointer_cast((int8_t *) data_left);
		int8_t int8_value = 2;
	//	thrust::fill(thrust::detail::make_normal_iterator(left_ptr), thrust::detail::make_normal_iterator(left_ptr + num_elements), int8_value);
		thrust::copy(thrust::make_counting_iterator<int8_t>(0),thrust::make_counting_iterator<int8_t>(0) + num_elements, thrust::detail::make_normal_iterator(left_ptr));

		thrust::device_ptr<int8_t> right_ptr= thrust::device_pointer_cast((int8_t *) data_right);
		int8_value = 2;
		thrust::fill(thrust::detail::make_normal_iterator(right_ptr), thrust::detail::make_normal_iterator(right_ptr + num_elements), int8_value);

		//for this simple test we will send in only 8 values
		gdf_valid_type * valid = new gdf_valid_type;


		*valid = 255;
		int num = 4;
		*valid = *valid & ~(1 << num);
		num = 6;
		*valid = *valid & ~(1 << num);

		//shold set lef tand bits 4 and 6 to off

		gdf_valid_type * valid_device;
		cuda_error = cudaMalloc((void **) &valid_device,64);
		cudaMemcpy(valid_device,valid,sizeof(gdf_valid_type),cudaMemcpyHostToDevice);
		gdf_valid_type * valid_out = new gdf_valid_type;
		cuda_error = cudaMalloc((void **) &valid_out,1);
		gdf_column lhs;
		gdf_error error = gdf_column_view(&lhs,(void *) data_left, valid_device,num_elements,GDF_INT8);
		lhs.null_count = 2;
		gdf_column rhs;
		error = gdf_column_view(&rhs,(void *) data_right, valid_device,num_elements,GDF_INT8);
		rhs.null_count = 2;
		gdf_column output;
		error = gdf_column_view(&output,(void *) data_out, valid_out,num_elements,GDF_INT8);

		std::cout<<"Left"<<std::endl;
		print_column(&lhs);
		std::cout<<"Right"<<std::endl;
		print_column(&rhs);

		error = gdf_binary_operation_v_v_v(&lhs,&rhs,&output,GDF_EQUAL);

		print_column(&output);

		error = gdf_binary_operation_v_v_v(&lhs,&rhs,&output,GDF_GREATER);

		print_column(&output);

		//copy the data on the host and compare
		thrust::device_ptr<int8_t> out_ptr = thrust::device_pointer_cast((int8_t *) output.data);
		
		gdf_data data = {.ui08=3};
		gdf_scalar three = {data, GDF_UINT8};

		error = gdf_binary_operation_v_s_v(&lhs,&three,&output,GDF_EQUAL);
		error = gdf_binary_operation_v_v_v(&lhs,&rhs,&output,GDF_GREATER);

		print_column(&output);

		gpu_apply_stencil(&lhs, &output, &rhs);

		print_column(&rhs);

	//	cudaMemcpy(valid,output.valid,1,cudaMemcpyDeviceToHost);

		cudaFree(data_left);
		cudaFree(data_right);
		cudaFree(data_out);
		cudaFree(valid_device);
		cudaFree(valid_out);

		delete valid;
}*/

/*void runInterpreterTest(){
	std::string expression = "AND(=(*($0, $0), 1), =($1, 2))";
	expression = "=(=($1, $0), $0)";

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

	blazing_frame blzframe;
	blzframe.add_table(inputs);

	gdf_column * output = new gdf_column;
	create_gdf_column(output, GDF_INT8, num_values, nullptr, 1);

	gdf_column * temp = new gdf_column;
	create_gdf_column(temp, GDF_INT8, num_values, nullptr, 1);

	std::cout<<"Input 0 ==>"<<std::endl;
	print_column(inputs[0]);
	std::cout<<std::endl<<"Input 1 ==>"<<std::endl;
	print_column(inputs[1]);
	std::cout<<std::endl<<"Input 2 ==>"<<std::endl;
	print_column(inputs[2]);

	expression = ">($1, 5)";
	std::cout<<"evaluating "<<expression<<std::endl;
	std::cout<<std::endl<<"Output before ==>"<<std::endl;
	print_column(output);
	evaluate_expression(
			blzframe,
			expression,
			output,
			temp);

	std::cout<<std::endl<<"Output after ==>"<<std::endl;
	print_column(output);

	expression = "=(=($1, $0), $2)";
	std::cout<<"evaluating "<<expression<<std::endl;
	std::cout<<std::endl<<"Output before ==>"<<std::endl;
	print_column(output);
	evaluate_expression(
			blzframe,
			expression,
			output,
			temp);

	std::cout<<std::endl<<"Output after ==>"<<std::endl;
	print_column(output);

	delete output;
	delete temp;
	//needs to be able to call this with temp == null to indicate we need to know how much space to allocate for temp
}*/

/*void runParquetTest(){
	gdf_column left;
	gdf_column right;
	gdf_column third;

	std::vector<gdf_column *> inputs(3);
	inputs[0] = &left;
	inputs[1] = &right;
	inputs[2] = &third;

	size_t num_values = 100;

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

	gdf_column ** columns = new gdf_column*[1];
	columns[0] = &left;
	char * test = (char *)"tester";
	char ** tester = new char *[1];
	tester[0] = test;
	size_t * size = new size_t;
	*size = 100;
	//gdf_error err = read_parquet("/home/felipe/parquet-test.parquet",nullptr,tester,columns,size);
}*/

/*void runCalciteTest()
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
}*/

/*void testStencil(){


	gdf_column * a = new gdf_column;
	gdf_column * b = new gdf_column;
	gdf_column * c = new gdf_column;

	//gdf_column a,b,c;

	int8_t a_data[3] = { 1,2,3 };
	int8_t b_data[3] = { 1,1,0 };
	//create_gdf_column(&a, GDF_INT8, 8, a_data, 1);
	//create_gdf_column(&b, GDF_INT8, 8, b_data, 1);
	//create_gdf_column(&c, GDF_INT8, 8, nullptr, 1);

	create_gdf_column(a, GDF_INT8, 3, a_data, 1);
	create_gdf_column(b, GDF_INT8, 3, b_data, 1);
	create_gdf_column(c, GDF_INT8, 3, nullptr, 1);

//	print_column(&a);
//	print_column(&b);
//	print_column(&c);

	print_column(a);
	print_column(b);
	print_column(c);

//	gpu_apply_stencil(
//						&a,
//						&b,
//						&c
//				);

	gpu_apply_stencil(
						a,
						b,
						c
				);

	//	print_column(&a);
	//	print_column(&b);
	//	print_column(&c);

		print_column(a);
		print_column(b);
		print_column(c);
}*/

typedef uint64_t connection_id;

enum action_t{
  OPEN_CONNECTION,
  CLOSE_CONNECTION
};

enum status_t{
  SUCCESS,
  FAILED
};

struct ConnectionRequest
{
  connection_id connection;
  action_t action;
};

struct ConnectionResponse
{
  status_t action;
  char msg[128];
};

static std::uint8_t data[4096];

int main(void)
{
	blazingdb::protocol::UnixSocketConnection connection("/tmp/socket");
	blazingdb::protocol::Server server(connection);

	auto controller = [](const blazingdb::protocol::Buffer &requestBuffer)
		-> blazingdb::protocol::Buffer {
		const ConnectionRequest *cr = reinterpret_cast<const ConnectionRequest *>(requestBuffer.data());

		ConnectionResponse resp;

		try {
			result_set_repository::get_instance().remove_all_connection_tokens(cr->connection);
			resp.action = SUCCESS;
		}
		catch(const std::runtime_error& e) {
			std::cout<<"Error: "<<e.what()<<std::endl;
			resp.action = FAILED;
			strcpy(resp.msg, e.what());
		}

		std::memcpy(data, &resp, sizeof(ConnectionResponse));

		return blazingdb::protocol::Buffer(data, sizeof(ConnectionResponse));
	};

	server.handle(controller);

	//runOriginalTest();
	//runInterpreterTest();
	//runCalciteTest();

	//testStencil();

	return 0;
}
