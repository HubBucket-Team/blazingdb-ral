/*
 * JoinProcessor.cpp
 *
 *  Created on: Aug 7, 2018
 *      Author: felipe
 */

#include "LogicalFilter.h"
#include "CalciteExpressionParsing.h"
//#include <cub/cub.cuh>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "DataFrame.h"
#include <stack>

//based on calcites relational algebra
const std::string INNER_JOIN = "inner";
const std::string LEFT_JOIN = "left";
const std::string RIGHT_JOIN = "right";
const std::string OUTER_JOIN = "full";


//TODO: a temporary function until we make it so that reductionss can
//specify output type independently of input type
void sum_column_i8(gdf_column * column, gdf_size_type & result, size_t num_items){

	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	//cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, (int8_t *) column->data, &result, num_items);
	// Allocate temporary storage
	//cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run sum-reduction
	//cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, (int8_t *) column->data, &result, num_items);
	//cudaFree(d_temp_storage);
}

//TODO: temp function should be removed when joins output gdf_column
struct bit_mask_pack_op : public thrust::unary_function<int64_t,gdf_valid_type>
{
	__host__ __device__
		gdf_valid_type operator()(const int64_t expanded)
		{
			gdf_valid_type result = 0;
			for(int i = 0; i < GDF_VALID_BITSIZE; i++){
				// 0, 8, 16, ....,48,  56
				unsigned char byte = (expanded >> ( (GDF_VALID_BITSIZE - 1 - i )  * 8));
				result |= (byte & 1) << i;
			}
			return (result);
		}
};

//TODO: temp function that should be removed when joins output gdf_column
/*void gdf_join_result_type_to_gdf_column(
		gdf_join_result_type * output,
		gdf_column * left_indices,
		gdf_column * right_indices,
		cudaStream_t stream){*/
	//TODO: this is a temporary but dangerous solution
	// we will have to keep a reference to the result_type to ensure
	//that we don't lose data

	//set left_indcees and right_indices to the pointers to output
	/*size_t result_set_size =  gdf_join_result_size(output);
	left_indices->data =  gdf_join_result_data(output);
	left_indices->dtype = GDF_INT32;
	left_indices->size = result_set_size;

	right_indices->data =  ((int *) gdf_join_result_data(output)) + result_set_size;
	right_indices->dtype = GDF_INT32;
	right_indices->size = result_set_size;


	gdf_valid_type * valid_data;
	cudaMalloc((void **) &valid_data,sizeof(char) * ((result_set_size + 7) / 8));
	left_indices->valid = valid_data;
	right_indices->valid = valid_data;*/
	/*
	 * we should not need this since its being implemented in libgdf
	//TODO: if we can ensure that all algorithms cn handle nullptr
	//bitmasks we could make our code more memory efficient here
	//createa  null bitmask for all values that are -1
	gdf_column * temp_stencil;
		//align ptr to 64 bytes as is specified in the arrow spec
	size_t num_bytes_valid = ((((result_set_size + 7) / 8) + 63) / 64) * 64;
	cudaMalloc((void **)&valid_ptr_left, num_bytes_valid );
	cudaMalloc((void **)&valid_ptr_right, num_bytes_valid);

	left_indices->valid = valid_ptr_left;
	right_indices->valid = valid_ptr_right;
	gdf_error err = gdf_column_view_augmented(temp_stencil,data,valid_ptr,
			result_set_size,GDF_INT8,0);
	all_bitmask_on(valid_ptr, & left_indices->null_count, result_set_size, stream);
	right_indices->null_count = 0;
	temp_stencil->null_count = 0;

	//gdf comparison to get stencil

	err = gpu_comparison_static_i32(left_indices,-1, temp_stencil,GDF_EQUALS);
	//sum to get count
	sum_column_i8(temp_stencil, left_indices->null_count, result_set_size);
	//convert stencil to bitmask


	//TODO: add this as a helper function to libgdf
	thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > valid_bit_mask_group_8_iter =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) temp_stencil->data));

	thrust::transform(thrust::cuda::par.on(stream), valid_bit_mask_group_8_iter, valid_bit_mask_group_8_iter + ((result_set_size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE),
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast(left_indices->valid)),bit_mask_pack_op());

	err = gpu_comparison_static_i32(right_indices,-1, temp_stencil,GDF_EQUALS);

	sum_column_i8(temp_stencil, right_indices->null_count, result_set_size);


	thrust::transform(thrust::cuda::par.on(stream), valid_bit_mask_group_8_iter, valid_bit_mask_group_8_iter + ((result_set_size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE),
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast(right_indices->valid)),bit_mask_pack_op());
	*/
//}

void evaluate_join(std::string condition,
		std::string join_type,
		blazing_frame data_frame,
		gdf_column * left_result,
		gdf_column * right_result
){
	
	/*gdf_column left_result;
	gdf_column right_result;*/
	//TODO: right now this only works for equijoins
	// since this is all that is implemented at the time


	std::string clean_expression = clean_calcite_expression(condition);
	int position = clean_expression.size();

	std::stack<std::string> operand;

	//TODO: for this to work properly we can only do multi column join
	// when we have ands, when we have hors we hvae to perform the joisn seperately then
	// do a unique merge of the indices


	//right now with pred push down the join codnition takes the filters as the second argument to condition

	int operator_count = 0;

	while(position > 0){
		std::string token = get_last_token(clean_expression,&position);
		//std::cout<<"Token is ==> "<<token<<"\n";

		if(is_operator_token(token)){

			if(token == "="){
				//so far only equijoins are supported in libgdf
				operator_count++;
			}else if(token != "AND"){
				throw std::runtime_error("In evaluate_join function: unsupported non-equijoins operator");
			}
		}else{
			operand.push(token);
		}
	}

	if(operator_count > 3 && join_type == OUTER_JOIN){
		throw std::runtime_error("In evaluate_join function: too many columns for join");
	}

	gdf_column ** left_columns = new gdf_column*[operator_count];
	gdf_column ** right_columns = new gdf_column*[operator_count];
	gdf_context ctxt{0, GDF_HASH, 0};
	int join_cols[operator_count];
	for(int i = 0; i < operator_count; i++){
		join_cols[i] = i;
		int right_index = get_index(operand.top());
		operand.pop();
		int left_index = get_index(operand.top());
		operand.pop();

		if(right_index < left_index){
			int temp_index = left_index;
			left_index = right_index;
			right_index = temp_index;
		}

		left_columns[i] = data_frame.get_column(left_index).get_gdf_column();
		right_columns[i] = data_frame.get_column(right_index).get_gdf_column();

	}

	if(join_type == INNER_JOIN){
		CUDF_CALL( gdf_inner_join( left_columns,operator_count,join_cols, right_columns,operator_count,join_cols,operator_count,0, nullptr,left_result, right_result, &ctxt) );

		/*
			* gdf_error gdf_inner_join(
						gdf_column **left_cols,
						int num_left_cols,
						int left_join_cols[],
						gdf_column **right_cols,
						int num_right_cols,
						int right_join_cols[],
						int num_cols_to_join,
						int result_num_cols,
						gdf_column **result_cols,
						gdf_column * left_indices,
						gdf_column * right_indices,
						gdf_context *join_context);

			*
			*//* --------------------------------------------------------------------------*/
		/**
		 * @Synopsis  Joins two dataframes (left, right) together on the specified columns
		 *
		 * @Param[in] left_cols[] The columns of the left dataframe
		 * @Param[in] num_left_cols The number of columns in the left dataframe
		 * @Param[in] left_join_cols[] The column indices of columns from the left dataframe
		 * to join on
		 * @Param[in] right_cols[] The columns of the right dataframe
		 * @Param[in] num_right_cols The number of columns in the right dataframe
		 * @Param[in] right_join_cols[] The column indices of columns from the right dataframe
		 * to join on
		 * @Param[in] num_cols_to_join The total number of columns to join on
		 * @Param[in] result_num_cols The number of columns in the resulting dataframe
		 * @Param[out] gdf_column *result_cols[] If not nullptr, the dataframe that results from joining
		 * the left and right tables on the specified columns
		 * @Param[out] gdf_column * left_indices If not nullptr, indices of rows from the left table that match rows in the right table
		 * @Param[out] gdf_column * right_indices If not nullptr, indices of rows from the right table that match rows in the left table
		 * @Param[in] join_context The context to use to control how the join is performed,e.g.,
		 * sort vs hash based implementation
		 *
		 * @Returns
		 */
	}else if(join_type == LEFT_JOIN){
//			err = gdf_left_join(operator_count, left_columns, right_columns, left_result, right_result, &ctxt);
		CUDF_CALL( gdf_left_join( left_columns,operator_count,join_cols, right_columns,operator_count,join_cols,operator_count,0, nullptr,left_result, right_result, &ctxt) );

	}else if(join_type == OUTER_JOIN){

	//WARNING TODO felipe percy noboa alexander see outer_join
	//err = gdf_outer_join_generic(left_columns[0], right_columns[0], left_result, right_result);

//			err = gdf_outer_join( left_columns,operator_count,join_cols, right_columns,operator_count,join_cols,operator_count,0, nullptr,left_result, right_result, &ctxt);
		CUDF_CALL( gdf_full_join(left_columns,operator_count,join_cols, right_columns,operator_count,join_cols,operator_count,0, nullptr,left_result, right_result, &ctxt) );
	}else{
		throw std::runtime_error("In evaluate_join function: unsupported join operator, " + join_type);
	}

	delete[] left_columns;
	delete[] right_columns;

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	//ya no gdf_join_result_type_to_gdf_column(output,left_indices,right_indices,stream);
	cudaStreamDestroy(stream);
}
