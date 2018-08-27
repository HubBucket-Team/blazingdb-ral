/*
 * LogicalFilter.cpp
 *
 *  Created on: Jul 20, 2018
 *      Author: felipe
 */

#include "LogicalFilter.h"

#include <stack>
#include <iostream>

#include "CalciteExpressionParsing.h"

//TODO: we need to update this to binary_operator
//when we have built that enum
gdf_error process__binary_operation_column_column(
		std::string operator_string,
		std::stack<std::string> & operands,
		std::vector<gdf_column *> & inputs,
		gdf_column * final_output,
		gdf_column * temp,
		bool is_last //set to true if we write to output
){
	gdf_column * output;
	if(is_last){
		output = final_output;
	}else{
		output = temp;
	}
	gdf_binary_operator operation;
	gdf_error err = get_operation(operator_string,&operation);
	if(err != GDF_SUCCESS){
		return err;
	}

	std::string left_operand = operands.top();
	operands.pop();
	std::string right_operand = operands.top();
	operands.pop();

	if(is_literal(left_operand)){
		if(is_literal(right_operand)){
			//kind of silly, should evalute literals
			//then copy results to output
		}else{
			//otro caso inverso
		}
	}else{
		size_t left_index = get_index(left_operand);

		if(is_literal(right_operand)){
			//TODO:libgdf should have an easier interface to passing in literals, like passing in a void * of the right
			//data type gpu_comparison_literal(same as gpu_comparison, void * literal)

			//for now we shortcut for our usecase
			//assuming type char
			if(inputs[left_index]->dtype == GDF_INT8){

				gdf_data data = {.ui32=stoi(right_operand)};
				gdf_scalar right = {data, GDF_UINT8};

				gdf_error err = gdf_binary_operation_v_s_v(inputs[left_index],&right,output,operation);
				if(err == GDF_SUCCESS){
					inputs.push_back(temp);
					operands.push("$" + std::to_string(inputs.size()-1));
				}
			}
		}else{

			size_t right_index = get_index(right_operand);

			gdf_error err = gdf_binary_operation_v_v_v(inputs[left_index],inputs[right_index],
					output,operation);
			if(err == GDF_SUCCESS){
				inputs.push_back(temp);
				operands.push("$" + std::to_string(inputs.size()-1));
			}
			return err;
		}
	}

}

template <typename T>
gdf_error process__binary_operation_column_literal(
		gdf_binary_operator operation,
		gdf_column * left,
		T right,
		gdf_column * output
){

}
template <typename T>
gdf_error process__binary_operation_literal_column(
		gdf_binary_operator operation,
		T left,
		gdf_column * right,
		gdf_column * output
){
	//TODO: only works for comparison operators

}



//processing in reverse we never need to have more than TWO spaces to work in
//
gdf_error evaluate_expression(
		std::vector<gdf_column *> inputs,
		std::string expression,
		gdf_column * output,
		gdf_column * temp){
	//make temp a column of size 8 bytes so it can accomodate the largest possible size

	std::string clean_expression = clean_calcite_expression(expression);
	int position = clean_expression.size();

	std::stack<std::string> operand_stack;


	while(position > 0){
		std::string token = get_last_token(clean_expression,&position);
		std::cout<<"Token is ==> "<<token<<"\n";

		if(is_operator_token(token)){
			process__binary_operation_column_column(
					token,
					operand_stack,
					inputs,
					output,
					temp,
					position == 0 ? true : false  //set to true if we write to output
			);


		}else{
			operand_stack.push(token);
		}
	}



	return GDF_SUCCESS;
}
