#include "interpreter_cpp.h"
#include "Interpreter/interpreter_ops.cuh"
#include "Config/Config.h"
#include "gdf_wrapper/gdf_wrapper.cuh"


const int THREAD_BLOCK_SIZE_8 = 256;
const int THREAD_BLOCK_SIZE_16 = THREAD_BLOCK_SIZE_8/2;
const int THREAD_BLOCK_SIZE_32 = THREAD_BLOCK_SIZE_8/4;
const int THREAD_BLOCK_SIZE_64 = THREAD_BLOCK_SIZE_8/8;


//TODO: a better way to handle all this thread block size is
//to get the amount of shared memory from the device and figure it out that way
//right now we are hardcoding to be able to handle pascal architecture cards


//we need to limit the number of threas per block depending on how mcuh shared memory we need per thread
typedef InterpreterFunctor<size_t,8,THREAD_BLOCK_SIZE_8> interpreter_functor_8;
typedef InterpreterFunctor<size_t,16,THREAD_BLOCK_SIZE_16> interpreter_functor_16;
typedef InterpreterFunctor<size_t,32,THREAD_BLOCK_SIZE_32> interpreter_functor_32;
typedef InterpreterFunctor<size_t,64,THREAD_BLOCK_SIZE_64> interpreter_functor_64;

/*
 *
 *
 *
 *
 *
 *
 dataframe input
 How to use this shit

 gdf_column col_input_1 = //make a column
 gdf_column col_input_2 = //make a column
 gdf_column col_input_3 = //make a column

 gdf_column col_output_1 = //allocate column
 gdf_column col_output_2 = //allocate column

 std::vector<gdf_column *> output_columns(2);
 output_columns[0] = &col_output_1;
 output_columns[1] = &col_output_2;

 std::vector<gdf_column *> input_columns(3);
 input_columns[0] = &col_input_1;
 input_columns[1] = &col_input_2;
 input_columns[2] = &col_input_3;


 temp_space starts at input.size + output.size = 5

 + * + $0 $1 $2 $1 , + sin $1 2.33   = step 0

 + * $5 $2 $1 , + $1 $2 step 1

 + $5 $1 , + $1 $2 step 2




Registers are
	0			1				2			3			4				5			6				n + 3 + 2
input_col_1, input_col_2, input_col_3, output_col_1, output_col2, processing_1, processing_2 .... processing_n

std::vector<column_index_type> & left_inputs = { 0, 5, 5, 1 ,5},
		std::vector<column_index_type> & right_inputs = { 1, 2, 1, -1, -2 },
		std::vector<column_index_type> & outputs { 5, 5, 3, 5,4 }


 std::vector<column_index_type> & final_output_positions = { 3 , 4 }

 std::vector<gdf_binary_operator> & operators = { GDF_ADD, GDF_MULT, GDF_ADD, GDF_INVALID_BINARY, GDF_ADD}
		std::vector<gdf_unary_operator> & unary_operators = { GDF_INVALID_UNARY,GDF_INVALID_UNARY,GDF_INVALID_UNARY,GDF_SIN,GDF_INVALID_UNARY  }

 		std::vector<gdf_scalar> & left_scalars = { junk, junk, junk, junk, junk }
		std::vector<gdf_scalar> & right_scalars = {junk, junk ,junk , 2.33, junk }

 		std::vector<column_index_type> new_input_indices = {0 , 1, 2 }

 perform_operation(all this shit you just made);
 hola alexander comoe estas
 *
 *
 */

gdf_error perform_operation(	std::vector<gdf_column *> output_columns,
std::vector<gdf_column *> input_columns,
std::vector<column_index_type> & left_inputs,
		std::vector<column_index_type> & right_inputs,
		std::vector<column_index_type> & outputs,
		std::vector<column_index_type> & final_output_positions,
		std::vector<gdf_binary_operator> & operators,
		std::vector<gdf_unary_operator> & unary_operators,


		std::vector<gdf_scalar> & left_scalars,
		std::vector<gdf_scalar> & right_scalars,
		std::vector<column_index_type> new_input_indices){


	//find maximum register used
	column_index_type max_output = 0;
	for(int i = 0; i < outputs.size(); i++){
		if(max_output < outputs[i]){
			max_output = outputs[i];
		}
	}


	char * temp_space;

	gdf_size_type num_rows = input_columns[0]->size;

	cudaStream_t stream;
	cudaStreamCreate(&stream);


	size_t shared_memory_per_thread = (max_output+1) * sizeof(int64_t);

	if(max_output <= 8){
		cudaMalloc(&temp_space,interpreter_functor_8::get_temp_size(input_columns.size(),left_inputs.size(),final_output_positions.size()));

		interpreter_functor_8 op(input_columns,
					output_columns,
					left_inputs.size(),
					left_inputs,
					right_inputs,
					outputs,
					final_output_positions,
					operators,
					unary_operators,
					left_scalars,
					right_scalars
					,stream,
					temp_space);
		transformKernel<<<32 * BlazingConfig::getInstance()->get_number_of_sms()
		,THREAD_BLOCK_SIZE_8,
	//	transformKernel<<<1
	//	,1,
						shared_memory_per_thread * THREAD_BLOCK_SIZE_8,
						stream>>>(op, num_rows);

		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
	}else if(max_output <= 16){
		cudaMalloc(&temp_space,interpreter_functor_16::get_temp_size(input_columns.size(),left_inputs.size(),final_output_positions.size()));

		interpreter_functor_16 op(input_columns,
					output_columns,
					left_inputs.size(),
					left_inputs,
					right_inputs,
					outputs,
					final_output_positions,
					operators,
					unary_operators,
					left_scalars,
					right_scalars,
					stream,
					temp_space);
		transformKernel<<<32 * BlazingConfig::getInstance()->get_number_of_sms()
				,THREAD_BLOCK_SIZE_16,
				shared_memory_per_thread * THREAD_BLOCK_SIZE_16,
				stream>>>(op, num_rows);
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);

	}else if(max_output <= 32){
		cudaMalloc(&temp_space,interpreter_functor_32::get_temp_size(input_columns.size(),left_inputs.size(),final_output_positions.size()));

		interpreter_functor_32 op(input_columns,
					output_columns,
					left_inputs.size(),
					left_inputs,
					right_inputs,
					outputs,
					final_output_positions,
					operators,
					unary_operators,
					left_scalars,
					right_scalars,
					stream,
					temp_space);

		transformKernel<<<32 * BlazingConfig::getInstance()->get_number_of_sms()
		,THREAD_BLOCK_SIZE_32,
						shared_memory_per_thread * THREAD_BLOCK_SIZE_32,
						stream>>>(op, num_rows);
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
	}else{ // if(max_output <= 64){
		cudaMalloc(&temp_space,interpreter_functor_64::get_temp_size(input_columns.size(),left_inputs.size(),final_output_positions.size()));

		interpreter_functor_64 op(input_columns,
					output_columns,
					left_inputs.size(),
					left_inputs,
					right_inputs,
					outputs,
					final_output_positions,
					operators,
					unary_operators,
					left_scalars,
					right_scalars,
					stream,
					temp_space);

		transformKernel<<<32 * BlazingConfig::getInstance()->get_number_of_sms()
		,THREAD_BLOCK_SIZE_64,
						shared_memory_per_thread * THREAD_BLOCK_SIZE_64,
						stream>>>(op, num_rows);
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
	}
	cudaFree(temp_space);


	return GDF_SUCCESS;
}
