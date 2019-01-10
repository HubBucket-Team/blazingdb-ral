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

#include "gdf_wrapper/gdf_wrapper.cuh"

//TODO: we need to update this to binary_operator
//when we have built that enum
gdf_error process__binary_operation_column_column(
		std::string operator_string,
		std::stack<std::string> & operands,
		blazing_frame & inputs,
		gdf_column_cpp final_output,
		gdf_column_cpp temp,
		bool is_last //set to true if we write to output
){
	gdf_column_cpp output;
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
			return GDF_INVALID_API_CALL;
		}else{
			//for now we shortcut for our usecase
			//assuming type char

			size_t right_index = get_index(right_operand);

			gdf_scalar left = get_scalar_from_string(left_operand,inputs.get_column(right_index).dtype());
			gdf_error err = gdf_binary_operation_v_s_v(output.get_gdf_column(),&left,inputs.get_column(right_index).get_gdf_column(),operation);
			if(err == GDF_SUCCESS){
				inputs.add_column(temp.clone());
				operands.push("$" + std::to_string(inputs.get_size_column()-1));
			}
			return err;
		}
	}else{
		size_t left_index = get_index(left_operand);

		if(is_literal(right_operand)){
			gdf_scalar right = get_scalar_from_string(right_operand,inputs.get_column(left_index).dtype());
			
			std::cout << ">>>>>>>>>>>> right " << (int)right.dtype << "  " << (right.data.si64) << std::endl;
			std::cout << ">>>>>>>>>>>> left" << std::endl;
			print_gdf_column(inputs.get_column(left_index).get_gdf_column());

			gdf_error err = gdf_binary_operation_v_v_s(output.get_gdf_column(),inputs.get_column(left_index).get_gdf_column(),&right,operation);
			std::cout << ">>>>>>>>>>>> out" << std::endl;
			print_gdf_column(output.get_gdf_column());
			if(err == GDF_SUCCESS){
				inputs.add_column(temp.clone());
				operands.push("$" + std::to_string(inputs.get_size_column()-1));
			}
			return err;
		}else{

			size_t right_index = get_index(right_operand);

			gdf_error err = gdf_binary_operation_v_v_v(output.get_gdf_column(),inputs.get_column(left_index).get_gdf_column(),inputs.get_column(right_index).get_gdf_column(),
					operation);
			if(err == GDF_SUCCESS){
				inputs.add_column(temp.clone());
				operands.push("$" + std::to_string(inputs.get_size_column()-1));
			}
			return err;
		}
	}

}

gdf_error process_unary_operation(
		std::string operator_string,
		std::stack<std::string> & operands,
		blazing_frame & inputs,
		gdf_column_cpp final_output,
		gdf_column_cpp temp,
		bool is_last //set to true if we write to output
){
	gdf_column_cpp output;
	if(is_last){
		output = final_output;
	}else{
		output = temp;
	}

	gdf_unary_operator operation; //need to define this

	gdf_error err = get_operation(operator_string,&operation);
	if(err != GDF_SUCCESS){
		return err;
	}

	std::string left_operand = operands.top();
	operands.pop();

	if(is_literal(left_operand)){


		return GDF_INVALID_API_CALL;
	}else{
		size_t left_index = get_index(left_operand);

		gdf_error err;
		switch (operation){
		case GDF_FLOOR:
			err = gdf_floor_generic(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_CEIL:
			err = gdf_ceil_generic(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_SIN:
			err = gdf_sin_generic(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_COS:
			err = gdf_cos_generic(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_ASIN:
			err = gdf_asin_generic(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_ACOS:
			err = gdf_acos_generic(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_TAN:
			err = gdf_tan_generic(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_ATAN:
			err = gdf_atan_generic(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_LN:
			err = gdf_log_generic(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_YEAR:
			err = gdf_extract_datetime_year(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_MONTH:
			err = gdf_extract_datetime_month(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_DAY:
			err = gdf_extract_datetime_day(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_HOUR:
			err = gdf_extract_datetime_hour(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_MINUTE:
			err = gdf_extract_datetime_minute(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_SECOND:
			err = gdf_extract_datetime_second(inputs.get_column(left_index).get_gdf_column(), output.get_gdf_column());
			break;
		case GDF_LOG:
			err = GDF_INVALID_API_CALL;
			break;
		case GDF_ABS:
			err = GDF_INVALID_API_CALL;
			break;
		case GDF_NOT:
			err = GDF_INVALID_API_CALL;
			break;
		default:
			err = GDF_INVALID_API_CALL;
		}


		if(err == GDF_SUCCESS){
			inputs.add_column(temp.clone());
			operands.push("$" + std::to_string(inputs.get_size_column()-1));
		}
		return err;
	}
}
 
gdf_error process_other_binary_operation(
		std::string operator_string,
		std::stack<std::string> & operands,
		blazing_frame & inputs,
		gdf_column_cpp final_output,
		gdf_column_cpp temp,
		bool is_last //set to true if we write to output
){
	gdf_column_cpp output;
	if(is_last){
		output = final_output;
	}else{
		output = temp;
	}

	gdf_other_binary_operator operation; 
	gdf_error err = get_operation(operator_string,&operation);
	if(err != GDF_SUCCESS){
		return err;
	}

	std::string left_operand = operands.top();
	operands.pop();
	std::string right_operand = operands.top();
	operands.pop();

	switch (operation){
		case GDF_COALESCE:
			if(is_literal(left_operand)){
				return GDF_INVALID_API_CALL;			
			} else {
				size_t left_index = get_index(left_operand);
				
				if(is_literal(right_operand)){
					// take literal and put into a size 1 column and call replace_nulls
					gdf_scalar right = get_scalar_from_string(right_operand, inputs.get_column(left_index).dtype());
					std::cout << ">>>>>>>>>>>> coalesce right " << right_operand << " "<<  (int)right.dtype << "  " << (right.data.si32) << std::endl;

					gdf_column_cpp expression_input = inputs.get_column(left_index);
					std::cout << ">>>>>>>>>>>> coalesce expression_input" << std::endl;
					print_gdf_column(expression_input.get_gdf_column());

					gdf_column_cpp temp_scalar;
					temp_scalar.create_gdf_column(right.dtype, 1, nullptr, get_width_dtype(right.dtype));
					CheckCudaErrors(cudaMemcpy(temp_scalar.data(), &(right.data), get_width_dtype(right.dtype), cudaMemcpyHostToDevice));

					err = gdf_replace_nulls(output.get_gdf_column(), expression_input.get_gdf_column(), temp_scalar.get_gdf_column());
					std::cout << ">>>>>>>>>>>> coalesce out" << std::endl;
					print_gdf_column(output.get_gdf_column());
				} else {
					// call replace_null
					size_t right_index = get_index(right_operand);

					err = gdf_replace_nulls(output.get_gdf_column(), inputs.get_column(left_index).get_gdf_column(), inputs.get_column(right_index).get_gdf_column());
				}
				if(err == GDF_SUCCESS){
					inputs.add_column(temp.clone());
					operands.push("$" + std::to_string(inputs.get_size_column()-1));
				}
			}
			break;
		default:
			err = GDF_INVALID_API_CALL;
	}
	
	return err;
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
)
{
	//TODO: only works for comparison operators
	
}



//processing in reverse we never need to have more than TWO spaces to work in
//
gdf_error evaluate_expression(
		blazing_frame inputs,
		std::string expression,
		gdf_column_cpp output,
		gdf_column_cpp temp){
	//make temp a column of size 8 bytes so it can accomodate the largest possible size

	std::string clean_expression = clean_calcite_expression(expression);
	int position = clean_expression.size();

	std::stack<std::string> operand_stack;


	while(position > 0){
		std::string token = get_last_token(clean_expression,&position);
		//std::cout<<"Token is ==> "<<token<<"\n";

		if(is_operator_token(token)){

			//Todo: Check correctness
			/*if (token == "OR") {
				auto op1 = operand_stack.top();
				operand_stack.pop();
				auto op2 = operand_stack.top();
				operand_stack.pop();

				std::string equal = { "MOD + " };
				equal.append(op1 + " " + op2 + " 2");
				clean_expression.erase(position, 2);
				clean_expression.insert(position, equal);
				position += equal.length();
				continue;
			}*/

			if(is_binary_operator_token(token)){
				process__binary_operation_column_column(
						token,
						operand_stack,
						inputs,
						output,
						temp,
						position == 0 ? true : false  //set to true if we write to output
				);

			}else if(is_unary_operator_token(token)){
				process_unary_operation(
						token,
						operand_stack,
						inputs,
						output,
						temp,
						position == 0 ? true : false  //set to true if we write to output
				);
			} else if (is_other_binary_operator_token(token)){
				process_other_binary_operation(
						token,
						operand_stack,
						inputs,
						output,
						temp,
						position == 0 ? true : false  //set to true if we write to output
				);
			} else {
				return GDF_INVALID_API_CALL;
			}


		}else{
			operand_stack.push(token);
		}
	}

	output.update_null_count();

	std::cout << "<<<<<<<<<<<<<<<<,,>>>>>>>>>>>> out" << std::endl;
	print_gdf_column(output.get_gdf_column());
	
	return GDF_SUCCESS;
}
