#include "interpreter_cpp.h"
#include "Interpreter/interpreter_ops.cuh"
#include "Config/Config.h"
#include "gdf_wrapper/gdf_wrapper.cuh"


const int THREAD_BLOCK_SIZE_8 = 512;
const int THREAD_BLOCK_SIZE_16 = 512/2;
const int THREAD_BLOCK_SIZE_32 = 512/4;
const int THREAD_BLOCK_SIZE_64 = 512/8;


//TODO: a better way to handle all this thread block size is
//to get the amount of shared memory from the device and figure it out that way
//right now we are hardcoding to be able to handle pascal architecture cards


//we need to limit the number of threas per block depending on how mcuh shared memory we need per thread
typedef InterpreterFunctor<size_t,8,THREAD_BLOCK_SIZE_8> interpreter_functor_8;
typedef InterpreterFunctor<size_t,16,THREAD_BLOCK_SIZE_16> interpreter_functor_16;
typedef InterpreterFunctor<size_t,32,THREAD_BLOCK_SIZE_32> interpreter_functor_32;
typedef InterpreterFunctor<size_t,64,THREAD_BLOCK_SIZE_64> interpreter_functor_64;

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
	gdf_size_type num_rows = input_columns[0]->size;

	cudaStream_t stream;
	cudaStreamCreate(&stream);


	size_t shared_memory_per_thread = max_output * sizeof(int64_t);

	if(max_output <= 8){
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
					,stream);

		transformKernel<<<32 * BlazingConfig::getInstance()->get_number_of_sms()
		,THREAD_BLOCK_SIZE_8,
						shared_memory_per_thread * THREAD_BLOCK_SIZE_8,
						stream>>>(op, num_rows);

	}else if(max_output <= 16){
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
					stream);

		transformKernel<<<32 * BlazingConfig::getInstance()->get_number_of_sms()
				,THREAD_BLOCK_SIZE_16,
				shared_memory_per_thread * THREAD_BLOCK_SIZE_16,
				stream>>>(op, num_rows);

	}else if(max_output <= 32){
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
					stream);

		transformKernel<<<32 * BlazingConfig::getInstance()->get_number_of_sms()
		,THREAD_BLOCK_SIZE_32,
						shared_memory_per_thread * THREAD_BLOCK_SIZE_32,
						stream>>>(op, num_rows);
	}else if(max_output <= 64){
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
					stream);

		transformKernel<<<32 * BlazingConfig::getInstance()->get_number_of_sms()
		,THREAD_BLOCK_SIZE_64,
						shared_memory_per_thread * THREAD_BLOCK_SIZE_64,
						stream>>>(op, num_rows);
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	return GDF_SUCCESS;
}
