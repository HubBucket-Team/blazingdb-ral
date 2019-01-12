
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <map>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include "gdf_wrapper/gdf_wrapper.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#include "CalciteExpressionParsing.h"

bool is_exponential_operator(gdf_binary_operator operation){
	return operation == GDF_POW;
}

/*
template<typename T>
__device__ __forceinline__ T t(int thread_id,
                                                      int logical_index)
{
    return logical_index * THREADBLOCK_SIZE + thread_id;
}
 */

__device__ __forceinline__
bool isInt(gdf_dtype type){
	return (type == GDF_INT32) ||
			(type == GDF_INT64) ||
			(type == GDF_INT16) ||
			(type == GDF_INT8) ||
			(type == GDF_DATE32) ||
			(type == GDF_DATE64) ||
			(type == GDF_TIMESTAMP);
}


__device__ __forceinline__
bool isFloat(gdf_dtype type){
	return (type == GDF_FLOAT64) ||
			(type == GDF_FLOAT32);
}

__device__ __forceinline__
bool isUnsignedInt(gdf_dtype type){
	return false;
	/* Unsigned types are not currently supported in cudf
	return (type == GDF_UINT32) ||
			(type == GDF_UINT64) ||
			(type == GDF_UINT16) ||
			(type == GDF_UINT8);*/
}




typedef short column_index_type;

/**
 * every element that is stored in the local buffer is 8 bytes in local, so signed ints are cast to int64, unsigned to uint64, and floating points
 * are all doubles
 */

template <typename IndexT, int BufferSize, int ThreadBlockSize>
class InterpreterFunctor {
private:
	void  **column_data; //these are device side pointers to the device pointer found in gdf_column.data
	void ** output_data;
	gdf_valid_type ** valid_ptrs; //device
	gdf_valid_type ** valid_ptrs_out;
	size_t num_columns;
	gdf_dtype * input_column_types;
	size_t num_rows;
	column_index_type *  left_input_positions; //device
	column_index_type * right_input_positions; //device
	column_index_type * output_positions; //device
	column_index_type * final_output_positions; //should be same size as output_data, e.g. num_outputs
	short num_final_outputs;
	short num_operations;
	gdf_dtype * input_types_left; //device
	gdf_dtype * input_types_right; //device
	gdf_dtype * output_types; //device
	gdf_dtype * final_output_types; //size
	gdf_binary_operator * binary_operations; //device
	gdf_unary_operator * unary_operations; //device
	char * temp_space;

	template<typename LocalStorageTyzpe>
	__device__ __forceinline__
	LocalStorageType get_data_from_buffer(
			int64_t * buffer, //the local buffer which storse the information that is to be processed
			column_index_type position) //the position in the local buffer where this data needs to be written
	{
		//columns
		//	return *col_data;
		//	printf("data %d",*((LocalStorageType *) (buffer + position)));


		//lets chang eposition to not be in bytes

		return *((LocalStorageType *) (buffer + ((position * ThreadBlockSize) + threadIdx.x)));

		//return (col_data[row]);
	}

	template<typename LocalStorageType>
	__device__ __forceinline__
	void store_data_in_buffer(
			LocalStorageType data,
			int64_t * buffer,
			column_index_type position){
		*((LocalStorageType *) (buffer + ((position * ThreadBlockSize) + threadIdx.x))) = data;
	}

	template<typename ColType, typename LocalStorageType>
	__device__ __forceinline__
	void device_ptr_read_into_buffer(int col_index,
			const IndexT row,
			const void * const * columns,
			int64_t * buffer, //the local buffer which storse the information that is to be processed
			column_index_type position){
		const ColType* col_data = static_cast<const ColType*>((columns[col_index]));
		//	return *col_data;
		*((LocalStorageType *) (buffer + ((position * ThreadBlockSize) + threadIdx.x))) = (LocalStorageType) __ldg(((ColType *) &col_data[row]));
	}

	template<typename ColType, typename LocalStorageType>
	__device__ __forceinline__
	void device_ptr_write_from_buffer(
			const IndexT row,
			void * columns,
			int64_t * buffer, //the local buffer which storse the information that is to be processed
			column_index_type position){
		const LocalStorageType col_data = *((LocalStorageType *) (buffer + ((position * ThreadBlockSize) + threadIdx.x)));
		((ColType *) columns)[row] = (ColType) col_data;
	}


	__device__
	__forceinline__ void write_data(column_index_type cur_column, column_index_type cur_buffer,  int64_t * buffer,const size_t & row_index){
		gdf_dtype cur_type = this->final_output_types[cur_column];
		if(cur_type == GDF_INT8){
			device_ptr_write_from_buffer<int8_t,int64_t>(

					row_index,
					this->output_data[cur_column],
					buffer,
					cur_buffer);


		}else if(cur_type == GDF_INT16){
			device_ptr_write_from_buffer<int16_t,int64_t>(

					row_index,
					this->output_data[cur_column],
					buffer,
					cur_buffer);


		}else if(cur_type == GDF_INT32 ||
				cur_type == GDF_DATE32){
			device_ptr_write_from_buffer<int32_t,int64_t>(

					row_index,
					this->output_data[cur_column],
					buffer,
					cur_buffer);


		}else if(cur_type == GDF_INT64 ||
				cur_type == GDF_DATE64 ||
				cur_type == GDF_TIMESTAMP){
			device_ptr_write_from_buffer<int64_t,int64_t>(

					row_index,
					this->output_data[cur_column],
					buffer,
					cur_buffer);


		}else if(cur_type == GDF_FLOAT32){
			device_ptr_write_from_buffer<float,double>(

					row_index,
					this->output_data[cur_column],
					buffer,
					cur_buffer);


		}else if(cur_type == GDF_FLOAT64){
			device_ptr_write_from_buffer<double,double>(

					row_index,
					this->output_data[cur_column],
					buffer,
					cur_buffer);


		}

	}

	//TODO: a clever person would make this something that gets passed into this function so that we can do more clever things than just read
	//data from some grumpy old buffer, like read in from another one of these that does read from a boring buffer, or from some perumtation iterartor
	//hmm in fact if it could read permuted data you could avoi dmaterializing intermeidate filter steps
	__device__
	__forceinline__ void read_data(column_index_type cur_column,  int64_t * buffer,const size_t & row_index){
		gdf_dtype cur_type = this->input_column_types[cur_column];

		if(cur_type == GDF_INT8){
			device_ptr_read_into_buffer<int8_t,int64_t>(
					cur_column,
					row_index,
					this->column_data,
					buffer,
					cur_column);


		}else if(cur_type == GDF_INT16){
			device_ptr_read_into_buffer<int16_t,int64_t>(
					cur_column,
					row_index,
					this->column_data,
					buffer,
					cur_column);


		}else if(cur_type == GDF_INT32 ||
				cur_type == GDF_DATE32){
			device_ptr_read_into_buffer<int32_t,int64_t>(
					cur_column,
					row_index,
					this->column_data,
					buffer,
					cur_column);


		}else if(cur_type == GDF_INT64 ||
				cur_type == GDF_DATE64 ||
				cur_type == GDF_TIMESTAMP){
			device_ptr_read_into_buffer<int64_t,int64_t>(
					cur_column,
					row_index,
					this->column_data,
					buffer,
					cur_column);


		}else if(cur_type == GDF_FLOAT32){
			device_ptr_read_into_buffer<float,double>(
					cur_column,
					row_index,
					this->column_data,
					buffer,
					cur_column);


		}else if(cur_type == GDF_FLOAT64){
			device_ptr_read_into_buffer<double,double>(
					cur_column,
					row_index,
					this->column_data,
					buffer,
					cur_column);


		}
	}

	/*
	__device__
	__forceinline__ void read_permuted_data(column_index_type cur_column,  int64_t * buffer,const size_t & row_index){
		//put permuted data here
	}
	 */
	__device__
	__forceinline__ void process_operator(size_t op_index,  int64_t * buffer){
		gdf_dtype type = this->input_types_left[op_index];
		if(isInt(type)){
			process_operator_1<int64_t>(op_index,buffer);
		}else if(isFloat(type)){
			process_operator_1<double>(op_index,buffer);
		}else if(isUnsignedInt(type)){
			process_operator_1<uint64_t>(op_index,buffer);
		}
	}

	template<typename LeftType>
	__device__
	__forceinline__ void process_operator_1(size_t op_index,  int64_t * buffer){
		gdf_dtype type = this->input_types_right[op_index];
		if(isInt(type)){
			process_operator_2<LeftType,int64_t>(op_index,buffer);
		}else if(isFloat(type)){
			process_operator_2<LeftType,double>(op_index,buffer);
		}else if(isUnsignedInt(type)){
			process_operator_2<LeftType,uint64_t>(op_index,buffer);
		}
	}

	template<typename LeftType, typename RightType>
	__device__
	__forceinline__ void process_operator_2(size_t op_index,  int64_t * buffer){
		gdf_dtype type = this->output_types[op_index];
		if(isInt(type)){
			process_operator_3<LeftType,RightType,int64_t>(op_index,buffer);
		}else if(isFloat(type)){
			process_operator_3<LeftType,RightType,double>(op_index,buffer);
		}else if(isUnsignedInt(type)){
			process_operator_3<LeftType,RightType,uint64_t>(op_index,buffer);
		}
	}

	template<typename LeftType, typename RightType, typename OutputTypeOperator>
	__device__
	__forceinline__ void process_operator_3(size_t op_index,  int64_t * buffer){

		column_index_type right_position = this->right_input_positions[op_index];
		column_index_type left_position = this->left_input_positions[op_index];

		column_index_type output_position = this->output_positions[op_index];
		if(right_position != -1){
			//binary op
			gdf_binary_operator oper = this->binary_operations[op_index];
			if(oper == GDF_ADD){
				store_data_in_buffer<OutputTypeOperator>(
						get_data_from_buffer<LeftType>(buffer,left_position)
						+ get_data_from_buffer<RightType>(buffer,right_position),
						buffer,
						output_position);
			}else if(oper == GDF_SUB){
				store_data_in_buffer<OutputTypeOperator>(
						get_data_from_buffer<LeftType>(buffer,left_position)
						- get_data_from_buffer<RightType>(buffer,right_position),
						buffer,
						output_position);
			}else if(oper == GDF_MUL){
				store_data_in_buffer<OutputTypeOperator>(
						get_data_from_buffer<LeftType>(buffer,left_position)
						* get_data_from_buffer<RightType>(buffer,right_position),
						buffer,
						output_position);
			}else if(oper == GDF_DIV){
				store_data_in_buffer<OutputTypeOperator>(
						get_data_from_buffer<LeftType>(buffer,left_position)
						/ get_data_from_buffer<RightType>(buffer,right_position),
						buffer,
						output_position);
			}else if(oper == GDF_TRUE_DIV){
				//TODO: snap this requires understanding of the bitmask
			}else if(oper == GDF_FLOOR_DIV){
				//TODO: figure out the same
			}else if(oper == GDF_MOD){
				//mod only makes sense with integer inputs
				store_data_in_buffer<OutputTypeOperator>(
						(int64_t) get_data_from_buffer<LeftType>(buffer,left_position)
						% (int64_t) get_data_from_buffer<RightType>(buffer,right_position),
						buffer,
						output_position);
			}else if(oper == GDF_POW){
				//oh god this is where it breaks if we are floats e do one thing
				OutputTypeOperator data = 1;
				if(isFloat((gdf_dtype) __ldg((int32_t *) &this->input_types_left[op_index])) || isFloat((gdf_dtype) __ldg((int32_t *) &this->input_types_right[op_index]))){
					data = pow((double) get_data_from_buffer<LeftType>(buffer,left_position),
							(double) get_data_from_buffer<RightType>(buffer,right_position));

				}else{
					//there is no pow for ints, so lets just do it...

					LeftType base = get_data_from_buffer<LeftType>(buffer,left_position);
					//right type is the exponent
					for(int i = 0; i < get_data_from_buffer<RightType>(buffer,right_position); i++){
						data *= base;
					}
				}
				store_data_in_buffer<OutputTypeOperator>(
						data,
						buffer,
						output_position);
			}else if(oper == GDF_EQUAL){
				store_data_in_buffer<OutputTypeOperator>(
						get_data_from_buffer<LeftType>(buffer,left_position)
						== get_data_from_buffer<RightType>(buffer,right_position),
						buffer,
						output_position);
			}else if(oper == GDF_NOT_EQUAL){
				store_data_in_buffer<OutputTypeOperator>(
						get_data_from_buffer<LeftType>(buffer,left_position)
						!= get_data_from_buffer<RightType>(buffer,right_position),
						buffer,
						output_position);
			}else if(oper == GDF_LESS){
				store_data_in_buffer<OutputTypeOperator>(
						get_data_from_buffer<LeftType>(buffer,left_position)
						< get_data_from_buffer<RightType>(buffer,right_position),
						buffer,
						output_position);
			}else if(oper == GDF_GREATER){
				store_data_in_buffer<OutputTypeOperator>(
						get_data_from_buffer<LeftType>(buffer,left_position)
						> get_data_from_buffer<RightType>(buffer,right_position),
						buffer,
						output_position);
			}else if(oper == GDF_LESS_EQUAL){
				store_data_in_buffer<OutputTypeOperator>(
						get_data_from_buffer<LeftType>(buffer,left_position)
						<= get_data_from_buffer<RightType>(buffer,right_position),
						buffer,
						output_position);
			}else if(oper == GDF_GREATER_EQUAL){
				store_data_in_buffer<OutputTypeOperator>(
						get_data_from_buffer<LeftType>(buffer,left_position)
						>= get_data_from_buffer<RightType>(buffer,right_position),
						buffer,
						output_position);
			}
		}else{
			//unary op
			gdf_unary_operator oper = this->unary_operations[op_index];

		}


	}

public:

	/*
	 * void  **column_data; //these are device side pointers to the device pointer found in gdf_column.data
	void ** output_data;
	gdf_valid_type ** valid_ptrs; //device
	gdf_valid_type ** valid_ptrs_out;
	size_t num_columns;
	gdf_dtype * input_column_types;
	size_t num_rows;
	column_index_type *  left_input_positions; //device
	column_index_type * right_input_positions; //device
	column_index_type * output_positions; //device
	column_index_type * final_output_positions; //should be same size as output_data, e.g. num_outputs
	short num_final_outputs;
	short num_operations;
	gdf_dtype * input_types_left; //device
	gdf_dtype * input_types_right; //device
	gdf_dtype * output_types; //device
	gdf_dtype * final_output_types; //size
	gdf_binary_operator * binary_operations; //device
	gdf_unary_operator * unary_operations;
	 */


	static size_t get_temp_size(std::vector<gdf_column> columns,
			short _num_operations,
			std::vector<short> left_input_positions_vec,
			std::vector<short> right_input_positions_vec,
			std::vector<short> output_positions_vec,
			std::vector<gdf_binary_operator> operators,
			short num_final_outputs){
		size_t space = 0;
		space += sizeof(void *) * columns.size(); //space for array of pointers to column data
		space += sizeof(void *) * num_final_outputs;
		space += sizeof(gdf_valid_type *) *columns.size(); //space for array of pointers  to column validity bitmasks
		space += sizeof(gdf_valid_type *) * num_final_outputs;
		space += sizeof(gdf_dtype) * columns.size(); //space for pointers to types for of input_columns
		space += 3 * (sizeof(short) * _num_operations); //space for pointers to indexes to columns e.g. left_input index, right_input index and output_index
		space += 3 * (sizeof(gdf_dtype) * _num_operations); //space for pointers to types for each input_index, e.g. if input_index = 1 then this value should contain column_1 type
		space += (sizeof(short) * num_final_outputs); //space for pointers to indexes to columns e.g. left_input index, right_input index and output_index
		space += (sizeof(gdf_dtype) * num_final_outputs); //space for pointers to types for each input_index, e.g. if input_index = 1 then this value should contain column_1 type
		space += sizeof(gdf_binary_operator) * _num_operations;
		space += sizeof(gdf_unary_operator) * _num_operations;

		return space;
	}
	//does not perform allocations
	InterpreterFunctor(void  ** column_data, //these are device side pointers to the device pointer found in gdf_column.data
			gdf_valid_type ** valid_ptrs, //device
			size_t num_columns,
			size_t num_rows,
			short * left_input_positions, //device
			short * right_input_positions, //device
			short * output_positions, //device
			short num_operations,
			gdf_dtype * input_types_left, //device
			gdf_dtype * input_types_right, //device
			gdf_dtype * output_types_out, //device
			gdf_binary_operator * binary_operations, //device
			gdf_dtype * input_column_types //device
	){
		this->column_data = column_data;
		this->valid_ptrs = valid_ptrs;
		this->num_columns = num_columns;
		this->num_rows = num_rows;
		this->left_input_positions = left_input_positions;
		this->right_input_positions = right_input_positions;
		this->output_positions = output_positions;
		this->num_operations = num_operations;
		this->input_types_left = input_types_left;
		this->input_types_right = input_types_right;
		this->output_types = output_types;
		this->binary_operations = binary_operations;
		this->input_column_types = input_column_types;
	}

	virtual ~InterpreterFunctor(){
		cudaFree(this->temp_space);
	}


	//simpler api and requires allocating just one block of temp space with
	// char * to make sure it can be dereferenced at one byte boundaries
	//This whole phase should take about ~ .1 ms, should
	//be using a stream for all this
	InterpreterFunctor(
			std::vector<gdf_column> columns,
			std::vector<gdf_column> output_columns,
			short _num_operations,
			std::vector<short> left_input_positions_vec,
			std::vector<short> right_input_positions_vec,
			std::vector<short> output_positions_vec,
			std::vector<short> final_output_positions_vec,
			std::vector<gdf_binary_operator> operators,
			std::vector<gdf_unary_operator> unary_operators//,

			//char * temp_space

	){


		this->num_final_outputs = final_output_positions_vec.size();
		this->num_operations = _num_operations;


		num_columns = columns.size();
		num_rows = columns[0].size;

		//added this to class
		//fuck this allocating is easier and i didnt see a significant differnece in tmie when i tried
		//to put this in constant memory
		size_t allocation_size = this->get_temp_size(columns,
				num_operations,
				left_input_positions_vec,
				right_input_positions_vec,
				output_positions_vec,
				operators,
				num_final_outputs);
		//		cudaGetSymbolAddress ( (void**)&cur_temp_space, shared_buffer);
		//		cudaGetSymbolAddress ( (void**)&column_data, shared_buffer);
		cudaMalloc(this->temp_space,allocation_size);
		char * cur_temp_space = this->temp_space;

		column_data = (void **) cur_temp_space;
		cur_temp_space += sizeof(void *) * num_columns;
		output_data = (void **) cur_temp_space;
		cur_temp_space += sizeof(void *) * num_final_outputs;
		valid_ptrs = (gdf_valid_type **) cur_temp_space;
		cur_temp_space += sizeof(gdf_valid_type *) * num_columns;
		valid_ptrs_out = (gdf_valid_type **) cur_temp_space;
		cur_temp_space += sizeof(gdf_valid_type *) * num_final_outputs;
		input_column_types = (gdf_dtype *) cur_temp_space;
		cur_temp_space += sizeof(gdf_dtype) * num_columns;
		left_input_positions = (short *) cur_temp_space;
		cur_temp_space += sizeof(short) * num_operations;
		right_input_positions = (short *) cur_temp_space;
		cur_temp_space += sizeof(short) * num_operations;
		output_positions = (short *) cur_temp_space;
		cur_temp_space += sizeof(short) * num_operations;
		final_output_positions = (short *) cur_temp_space;
		cur_temp_space += sizeof(short) * num_final_outputs;
		input_types_left = (gdf_dtype *) cur_temp_space;
		cur_temp_space += sizeof(gdf_dtype) * num_operations;
		input_types_right= (gdf_dtype *) cur_temp_space;
		cur_temp_space += sizeof(gdf_dtype) * num_operations;
		output_types = (gdf_dtype *) cur_temp_space;
		cur_temp_space += sizeof(gdf_dtype) * num_operations;
		final_output_types = (gdf_dtype *) cur_temp_space;
		cur_temp_space += sizeof(gdf_dtype) * num_final_outputs;
		binary_operations = (gdf_binary_operator *) cur_temp_space;
		cur_temp_space += sizeof(gdf_dtype) * num_final_outputs;
		binary_operations = (gdf_binary_operator *) cur_temp_space;
		cur_temp_space += sizeof(gdf_binary_operator) * num_final_outputs;
		unary_operations = (gdf_unary_operator *) cur_temp_space;


		std::vector<void *> host_data_ptrs(num_columns);
		std::vector<gdf_valid_type *> host_valid_ptrs(num_columns);
		for(int i = 0; i < num_columns; i++){
			host_data_ptrs[i] = columns[i].data;
			host_valid_ptrs[i] = columns[i].valid;
		}

		cudaError_t error = cudaMemcpy(this->column_data,&host_data_ptrs[0],sizeof(void *) * num_columns,cudaMemcpyHostToDevice);
		//	cudaError_t error = cudaMemcpy(this->column_data,&host_data_ptrs[0],sizeof(void *) * num_columns,cudaMemcpyHostToDevice);
		//	std::cout<<"about to copy host valid"<<error<<std::endl;
		//	error = cudaMemcpy(this->valid_ptrs,&host_valid_ptrs[0],sizeof(void *) * num_columns,cudaMemcpyHostToDevice);
		error = cudaMemcpy(this->valid_ptrs,&host_valid_ptrs[0],sizeof(void *) * num_columns,cudaMemcpyHostToDevice);

		//	std::cout<<"copied data and valid"<<error<<std::endl;


		host_data_ptrs.resize(num_final_outputs);
		host_valid_ptrs.resize(num_final_outputs);

		for(int i = 0; i < num_final_outputs; i++){
			host_data_ptrs[i] = output_columns[i].data;
			host_valid_ptrs[i] = output_columns[i].valid;
		}
		//	error = cudaMemcpy(this->output_data,&host_data_ptrs[0],sizeof(void *) * num_final_outputs,cudaMemcpyHostToDevice);

		error = cudaMemcpy(this->output_data,&host_data_ptrs[0],sizeof(void *) * num_final_outputs,cudaMemcpyHostToDevice);

		//	std::cout<<"about to copy host valid"<<error<<std::endl;
		//		error = cudaMemcpy(this->valid_ptrs_out,&host_valid_ptrs[0],sizeof(void *) * num_final_outputs,cudaMemcpyHostToDevice);
		error = cudaMemcpy(this->valid_ptrs_out,&host_valid_ptrs[0],sizeof(void *) * num_final_outputs,cudaMemcpyHostToDevice);


		//copy over inputs

		std::vector<gdf_dtype> left_input_types_vec(num_operations);
		std::vector<gdf_dtype> right_input_types_vec(num_operations);
		std::vector<gdf_dtype> output_types_vec(num_operations);
		std::vector<gdf_dtype> output_final_types_vec(num_final_outputs);




		//stores index to type map so we can retrieve this if we need to
		std::map<size_t,gdf_dtype> output_map_type;

		for(int cur_operation = 0; cur_operation < num_operations; cur_operation++){
			size_t left_index = left_input_positions_vec[cur_operation];
			size_t right_index = right_input_positions_vec[cur_operation];
			size_t output_index = output_positions_vec[cur_operation];

			if( left_index < columns.size()){
				left_input_types_vec[cur_operation] = columns[left_index].dtype;
			}else{
				//have to get it from the output that generated it
				left_input_types_vec[cur_operation] = output_map_type[left_index];
				//		std::cout<<"left type was "<<left_input_types_vec[cur_operation]<<std::endl;
			}

			if( right_index < columns.size()){
				right_input_types_vec[cur_operation] = columns[right_index].dtype;
			}else{
				right_input_types_vec[cur_operation] = output_map_type[right_index];
				//		std::cout<<"right type was "<<right_input_types_vec[cur_operation]<<std::endl;
			}

			gdf_dtype type_from_op = get_output_type(left_input_types_vec[cur_operation],
					right_input_types_vec[cur_operation],
					operators[cur_operation]);

			//		std::cout<<"type from op was "<<type_from_op<<std::endl;
			if(is_type_signed(type_from_op) && !(is_type_float(type_from_op))){
				output_types_vec[cur_operation] = GDF_INT64;
			}else if(is_type_float(type_from_op)){
				output_types_vec[cur_operation] = GDF_FLOAT64;
			}
			//		std::cout<<"op will be "<<output_types_vec[cur_operation]<<std::endl;


			output_map_type[output_index] = output_types_vec[cur_operation];

		}


		//put the output final positions
		for(int output_index = 0; output_index < num_final_outputs; output_index++){
			output_final_types_vec[output_index] = output_columns[output_index].dtype;
		}

		std::vector<gdf_dtype> input_column_types_vec(num_columns);
		for(int column_index = 0; column_index < columns.size(); column_index++){
			input_column_types_vec[column_index] = columns[column_index].dtype;
			//		std::cout<<"type was "<<input_column_types_vec[column_index]<<std::endl;
		}


		try{


			cudaMemcpy (left_input_positions,
					&left_input_positions_vec[0],
					left_input_positions_vec.size() * sizeof(short),cudaMemcpyHostToDevice);

			cudaMemcpy (right_input_positions,
					&right_input_positions_vec[0],
					right_input_positions_vec.size() * sizeof(short),cudaMemcpyHostToDevice);

			cudaMemcpy (output_positions,
					&output_positions_vec[0],
					output_positions_vec.size() * sizeof(short),cudaMemcpyHostToDevice);

			cudaMemcpy (final_output_positions,
					&final_output_positions_vec[0],
					final_output_positions_vec.size() * sizeof(short),cudaMemcpyHostToDevice);
			/*
			thrust::copy(left_input_positions_vec.begin(),left_input_positions_vec.end(),thrust::device_pointer_cast(left_input_positions));
			thrust::copy(right_input_positions_vec.begin(),right_input_positions_vec.end(),thrust::device_pointer_cast(right_input_positions));
			thrust::copy(output_positions_vec.begin(),output_positions_vec.end(),thrust::device_pointer_cast(output_positions));
			thrust::copy(final_output_positions_vec.begin(),final_output_positions_vec.end(),thrust::device_pointer_cast(final_output_positions));

			 */

			cudaMemcpy (binary_operations,
					&operators[0],
					operators.size() * sizeof(gdf_binary_operator),cudaMemcpyHostToDevice);

			cudaMemcpy (unary_operations,
					&unary_operators[0],
					unary_operators.size() * sizeof(gdf_unary_operator),cudaMemcpyHostToDevice);

			//	thrust::copy(operators.begin(),operators.end(),thrust::device_pointer_cast(binary_operations));

			cudaMemcpy (input_column_types,
					&input_column_types_vec[0],
					input_column_types_vec.size() * sizeof(gdf_dtype),cudaMemcpyHostToDevice);


			//	thrust::copy(input_column_types_vec.begin(), input_column_types_vec.end(), thrust::device_pointer_cast(input_column_types));


			cudaMemcpy (input_types_left,
					&left_input_types_vec[0],
					left_input_types_vec.size() * sizeof(gdf_dtype),cudaMemcpyHostToDevice);

			cudaMemcpy (input_types_right,
					&right_input_types_vec[0],
					right_input_types_vec.size() * sizeof(gdf_dtype),cudaMemcpyHostToDevice);

			cudaMemcpy (output_types,
					&output_types_vec[0],
					output_types_vec.size() * sizeof(gdf_dtype),cudaMemcpyHostToDevice);

			cudaMemcpy (final_output_types,
					&output_final_types_vec[0],
					output_final_types_vec.size() * sizeof(gdf_dtype),cudaMemcpyHostToDevice);
			/*
			thrust::copy(left_input_types_vec.begin(),left_input_types_vec.end(), thrust::device_pointer_cast(input_types_left));
			thrust::copy(right_input_types_vec.begin(),right_input_types_vec.end(), thrust::device_pointer_cast(input_types_right));
			thrust::copy(output_types_vec.begin(),output_types_vec.end(), thrust::device_pointer_cast(output_types));
			thrust::copy(output_final_types_vec.begin(),output_final_types_vec.end(), thrust::device_pointer_cast(final_output_types));
			 */

			//			std::cout<<"copied data!"<<std::endl;
			cudaDeviceSynchronize();

		}catch( ...){
			std::cout<<"could not copy data!"<<std::endl;
		}



	}
	__device__ __forceinline__ void operator()(const IndexT &row_index) {
		//		__shared__ char buffer[BufferSize * THREADBLOCK_SIZE];
		__shared__  int64_t  total_buffer[BufferSize * ThreadBlockSize];



		//here we are basically upgrading our inputs to a larger type

		//		#pragma unroll
		for(short cur_column = 0; cur_column < this->num_columns; cur_column++ ){
			read_data(cur_column,total_buffer, row_index);
		}


		for(short op_index = 0; op_index < this->num_operations; op_index++ ){
			process_operator(op_index, total_buffer );
		}

		//		#pragma unroll
		for(int out_index = 0; out_index < this->num_final_outputs; out_index++ ){
			write_data(out_index,this->final_output_positions[out_index],total_buffer,row_index);
		}




	}
};




void delete_gdf_column(gdf_column col){
	cudaFree(col.data);
	cudaFree(col.valid);
}

gdf_column create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value)
{

	cudaError_t error;
	gdf_column column;
	char * data;
	gdf_valid_type * valid_device;

	size_t allocated_size_valid = ((((num_values + 7 ) / 8) + 63 ) / 64) * 64; //so allocations are supposed to be 64byte aligned
	std::cout<<"allocating valid "<<allocated_size_valid<<std::endl;
	error = cudaMalloc( &valid_device, allocated_size_valid);
	std::cout<<"allocated valid "<<allocated_size_valid<<std::endl;
	//	std::cout<<"allocated device valid"<<error<<std::endl;
	cudaMemset(valid_device, (gdf_valid_type)255, allocated_size_valid); //assume all relevant bits are set to on

	size_t allocated_size_data = (((width_per_value * num_values) + 63 )/64) * 64;
	std::cout<<"allocating data "<<allocated_size_data<<std::endl;
	cudaMalloc( &data, allocated_size_data);
	std::cout<<"allocated data "<<allocated_size_data<<std::endl;
	gdf_column_view(&column, (void *) data, valid_device, num_values, type);

	if(input_data != nullptr){

		cudaMemcpy(data, input_data, num_values * width_per_value, cudaMemcpyHostToDevice);
		//		std::cout<<"copied memory"<<error<<std::endl;
	}
	return column;

}


template<typename interpreted_operator>
__global__ void transformKernel(interpreted_operator op, size_t size)
{

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
			i < size;
			i += blockDim.x * gridDim.x)
	{
		op(i);
	}


	return;
}

/*
class Timer
{
public:
	Timer() : beg_(clock_::now()) {}
	void reset() { beg_ = clock_::now(); }
	int64_t elapsed() const {
		return (std::chrono::high_resolution_clock::now() - beg_).count();
	}
private:
	typedef std::chrono::high_resolution_clock clock_;

	std::chrono::time_point<clock_> beg_;
};


const int THREAD_BLOCK_SIZE = 512;
int main(void)
{


	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);


	printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
			prop.multiProcessorCount,
			_ConvertSMVer2Cores(prop.major, prop.minor),
			_ConvertSMVer2Cores(prop.major, prop.minor) *
			prop.multiProcessorCount);

	int number_of_sms = prop.multiProcessorCount;


	typedef InterpreterFunctor<size_t,8,THREAD_BLOCK_SIZE> interpreter_functor;
	size_t limit = 10000001; //60000000
	for(size_t num_rows = 10000000; num_rows < limit; num_rows += 10000000){
		std::cout<<std::endl<<std::endl<<std::endl<<"############################################################"<<std::endl;
		std::cout<<"Running with "<<num_rows<<" rows!"<<std::endl;
		std::cout<<"############################################################"<<std::endl;

		std::vector<gdf_column> columns(2);

		//make 2 gdf columns

		int* input_data = new int[num_rows];
		for(int i = 0; i < num_rows; i++){
			input_data[i] = i;
		}

		double* input_data_dbl = new double[num_rows];
		for(int i = 0; i < num_rows; i++){
			input_data_dbl[i] = ((double)i) + .5;
		}

		columns[0] = create_gdf_column(GDF_INT32, num_rows, (void *) input_data, 4);
		columns[1] = create_gdf_column(GDF_FLOAT64, num_rows, (void *) input_data_dbl, 8);

		std::cout<<"Created columns"<<std::endl;



		std::vector<gdf_column> output_columns(2);
		output_columns[0] = create_gdf_column(GDF_FLOAT64, num_rows, nullptr, 8);
		output_columns[1] = create_gdf_column(GDF_FLOAT64, num_rows, nullptr, 8);
		std::cout<<"created op no making a vector of size"<<num_rows<<std::endl;


		Timer timer;

		std::vector<short> left_inputs = { 0 , 2};
		std::vector<short> right_inputs = { 1,  1};
		std::vector<short> outputs { 2, 3 };
		std::vector<short> final_output_positions { 1, 3 };
		std::vector<gdf_binary_operator> operators = { GDF_ADD, GDF_MUL};
		//		char * temp_space;
		size_t num_final_outputs = 2;
		//	size_t temp_space_needed = interpreter_functor::get_temp_size(columns,2,left_inputs,right_inputs,outputs,operators,num_final_outputs);
		//	std::cout<<"Planning phase ==>"<<(double)timer.elapsed() / 1000000.0<<"ms"<<std::endl;
		//	std::cout<<"need this much temp space"<<temp_space_needed<<std::endl;
		//	cudaMalloc(&temp_space,temp_space_needed);
		//	std::cout<<"allocted temp space"<<std::endl;







		interpreter_functor op(columns,output_columns,2,left_inputs,right_inputs,outputs,final_output_positions,operators);


		thrust::counting_iterator<size_t> iota_iter(0);


		std::cout<<"Planning phase ==>"<<(double)timer.elapsed() / 1000000.0<<"ms"<<std::endl;
		//		thrust::for_each(iota_iter, iota_iter + num_rows, op);
		//thrust::transform(thrust::cuda::par,iota_iter, iota_iter + num_rows, thrust::device_pointer_cast((double *)output_column.data), op);
		cudaDeviceSynchronize();
		//		std::cout<<"Total Time==>"<<(double)timer.elapsed() / 1000000.0<<"ms"<<std::endl;


		transformKernel<<<32 * number_of_sms,THREAD_BLOCK_SIZE>>>(op, num_rows);
		cudaDeviceSynchronize();
		std::cout<<"Total Time kernel==>"<<(double)timer.elapsed() / 1000000.0<<"ms"<<std::endl;
		std::cout<<"column 0 from operator"<<std::endl;
		thrust::copy(thrust::device_pointer_cast((double *)output_columns[0].data),
				thrust::device_pointer_cast((double *)output_columns[0].data) + num_rows,
				data_out_host.begin());

		for(int i = 0; i < num_rows; i++){
			if(i == 1 || i == (num_rows - 1)){
				std::cout<<"Row "<<i<<" = "<<data_out_host[i]<<std::endl;
			}
		}

		std::cout<<"column 1 from operator"<<std::endl;
		thrust::copy(thrust::device_pointer_cast((double *)output_columns[1].data),
				thrust::device_pointer_cast((double *)output_columns[1].data) + num_rows,
				data_out_host.begin());

		for(int i = 0; i < num_rows; i++){
			if(i == 1 || i == (num_rows - 1)){
				std::cout<<"Row "<<i<<" = "<<data_out_host[i]<<std::endl;
			}

		}
		delete_gdf_column(output_columns[0]);
		delete_gdf_column(output_columns[1]);
		cudaDeviceSynchronize();
		timer.reset();
		gdf_column output_temp = create_gdf_column(GDF_FLOAT64, num_rows, nullptr, 8);
		gdf_column output_final = create_gdf_column(GDF_FLOAT64, num_rows, nullptr, 8);
		cudaDeviceSynchronize();
		std::cout<<"temp allocations ==>"<<(double)timer.elapsed() / 1000000.0<<"ms"<<std::endl;
		//	gdf_binary_operation_v_v_v(&output_temp, &columns[0], &columns[1],GDF_ADD);
		//	gdf_binary_operation_v_v_v(&output_final, &output_temp, &columns[1],GDF_MUL);

		thrust::transform(thrust::cuda::par,thrust::device_pointer_cast((int *) columns[0].data),
				thrust::device_pointer_cast((int *) columns[0].data) + num_rows,
				thrust::device_pointer_cast((double *) columns[1].data),
				thrust::device_pointer_cast((double *) output_temp.data),
				thrust::plus<double>());

		thrust::transform(thrust::cuda::par,thrust::device_pointer_cast((double *) output_temp.data),
				thrust::device_pointer_cast((double *) output_temp.data) + num_rows,
				thrust::device_pointer_cast((double *) columns[1].data),
				thrust::device_pointer_cast((double *) output_final.data),
				thrust::multiplies<double>());

		cudaDeviceSynchronize();
		std::cout<<"total time thrust ==>"<<(double)timer.elapsed() / 1000000.0<<"ms"<<std::endl;
		thrust::copy(thrust::device_pointer_cast((double *)output_final.data),
				thrust::device_pointer_cast((double *)output_final.data) + num_rows,
				data_out_host.begin());


		for(int i = 0; i < num_rows; i++){
			if(i == 1 || i == (num_rows - 1)){
				std::cout<<"thrust Row "<<i<<" = "<<data_out_host[i]<<std::endl;
			}

		}



		delete_gdf_column(columns[0]);
		delete_gdf_column(columns[1]);
		delete_gdf_column(output_temp);
		delete_gdf_column(output_final);

	}



	return 0;
}
*/
