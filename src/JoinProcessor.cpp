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

void evaluate_join(std::string condition,
		std::string join_type,
		blazing_frame data_frame,
		gdf_column * left_result,
		gdf_column * right_result
){
	std::string clean_expression = clean_calcite_expression(condition);
	
	std::stack<std::string> operand;

	//TODO: for this to work properly we can only do multi column join
	// when we have ands, when we have hors we hvae to perform the joisn seperately then
	// do a unique merge of the indices


	//right now with pred push down the join codnition takes the filters as the second argument to condition

	int operator_count = 0;

	std::vector<std::string> tokens = get_tokens_in_reverse_order(clean_expression);
	for (std::string token : tokens){
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
	}else if(join_type == LEFT_JOIN){
		CUDF_CALL( gdf_left_join( left_columns,operator_count,join_cols, right_columns,operator_count,join_cols,operator_count,0, nullptr,left_result, right_result, &ctxt) );
	}else if(join_type == OUTER_JOIN){
		CUDF_CALL( gdf_full_join(left_columns,operator_count,join_cols, right_columns,operator_count,join_cols,operator_count,0, nullptr,left_result, right_result, &ctxt) );
	}else{
		throw std::runtime_error("In evaluate_join function: unsupported join operator, " + join_type);
	}

	delete[] left_columns;
	delete[] right_columns;
}
