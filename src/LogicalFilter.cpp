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
#include <NVCategory.h>
#include <NVStrings.h>

#include <blazingdb/io/Library/Logging/Logger.h>
#include "CodeTimer.h"
#include "gdf_wrapper/gdf_wrapper.cuh"

#include "Interpreter/interpreter_cpp.h"

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
	gdf_binary_operator operation = get_binary_operation(operator_string);

	if(operands.size()<2)
		throw std::runtime_error("In function process__binary_operation_column_column, the operator cannot be processed on less than one or zero elements");

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

			gdf_error err = gdf_binary_operation_v_v_s(output.get_gdf_column(),inputs.get_column(left_index).get_gdf_column(),&right,operation);
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

	//need to define this
	gdf_unary_operator operation = get_unary_operation(operator_string);

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

void process_other_binary_operation(
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

	gdf_other_binary_operator operation = get_other_binary_operation(operator_string);

	if(operands.size()<2)
		throw std::runtime_error("In function process_other_binary_operation, the operator cannot be processed on less than one or zero elements");

	std::string left_operand = operands.top();
	operands.pop();
	std::string right_operand = operands.top();
	operands.pop();

	switch (operation){
	case GDF_COALESCE:
		if(is_literal(left_operand)){
			throw std::runtime_error("In function process_other_binary_operation: unsupported operand, " + left_operand);
		} else {
			size_t left_index = get_index(left_operand);

			if(is_literal(right_operand)){
				// take literal and put into a size 1 column and call replace_nulls
				gdf_scalar right = get_scalar_from_string(right_operand, inputs.get_column(left_index).dtype());

				gdf_column_cpp expression_input = inputs.get_column(left_index);

				gdf_column_cpp temp_scalar;
				temp_scalar.create_gdf_column(right.dtype, 1, nullptr, get_width_dtype(right.dtype));
				CheckCudaErrors(cudaMemcpy(temp_scalar.data(), &(right.data), get_width_dtype(right.dtype), cudaMemcpyHostToDevice));
				//TODO: this function isnt used anymore but where is this replace nulls anyway?
			//	err = gdf_replace_nulls(output.get_gdf_column(), expression_input.get_gdf_column(), temp_scalar.get_gdf_column());
			} else {
				// call replace_null
				size_t right_index = get_index(right_operand);
				//TODO: this function isnt used anymore but where is this replace nulls anyway?
	//			err = gdf_replace_nulls(output.get_gdf_column(), inputs.get_column(left_index).get_gdf_column(), inputs.get_column(right_index).get_gdf_column());
			}
			
			inputs.add_column(temp.clone());
			operands.push("$" + std::to_string(inputs.get_size_column()-1));
		}
		break;
	default:
		throw std::runtime_error("In function process_other_binary_operation: unsupported operation, " + operator_string);
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
)
{
	//TODO: only works for comparison operators

}


typedef struct {
	std::string token;
	column_index_type position;
} operand_position;

column_index_type get_first_open_position(std::vector<bool> & open_positions, column_index_type start_position){
	for(column_index_type index =  start_position;index < open_positions.size(); index++ ){
		if(open_positions[index]){
			open_positions[index] = false;
			return index;
		}
	}
	return -1;
}


/**
 * Creates a physical plan for the expression that can be added to the total plan
 */
void add_expression_to_plan(	blazing_frame & inputs,
		std::vector<gdf_column *>& input_columns, 
		std::string expression,
		column_index_type expression_position,
		column_index_type num_outputs,
		column_index_type num_inputs,
		std::vector<column_index_type> & left_inputs,
		std::vector<column_index_type> & right_inputs,
		std::vector<column_index_type> & outputs,

		std::vector<gdf_binary_operator> & operators,
		std::vector<gdf_unary_operator> & unary_operators,


		std::vector<gdf_scalar> & left_scalars,
		std::vector<gdf_scalar> & right_scalars,
		std::vector<column_index_type> & new_input_indices){

	/*
	 * inputs needed
	 * std::vector<gdf_column> columns,
			std::vector<gdf_column> output_columns,
			short _num_operations,
			std::vector<short> left_input_positions_vec,
			std::vector<short> right_input_positions_vec,
			std::vector<short> output_positions_vec,
			std::vector<short> final_output_positions_vec,
			std::vector<gdf_binary_operator> operators,
			std::vector<gdf_unary_operator> unary_operators,
			std::vector<gdf_scalar> left_scalars, //should be same size as operations with most of them filled in with invalid types unless scalar is used in oepration
			std::vector<gdf_scalar> right_scalars//,
	 */

	//handled in parent
	//std::vector<column_index_type> final_output_positions;
	//std::vector<gdf_column> output_columns;


	column_index_type start_processing_position = num_inputs + num_outputs;


	std::string clean_expression = clean_calcite_expression(expression);

	int position = clean_expression.size();


	std::stack<operand_position> operand_stack;
	gdf_scalar dummy_scalar;

	std::vector<bool> processing_space_free(512,true); //a place to stare whether or not a processing space is occupied at any point in time
	for(size_t i = 0; i < start_processing_position; i++){
		processing_space_free[i] = false;
	}
	//pretend they are like registers and we need to know how many registers we need to evaluate this expression

	while(position > 0){
		std::string token = get_last_token(clean_expression,&position);

		if(is_operator_token(token)){
			if(is_binary_operator_token(token)){

				std::string left_operand = operand_stack.top().token;
				if(!is_literal(left_operand)){
					if(operand_stack.top().position >= start_processing_position){
						processing_space_free[operand_stack.top().position] = true;
					}
				}
				operand_stack.pop();
				std::string right_operand = operand_stack.top().token;
				if(!is_literal(right_operand)){
					if(operand_stack.top().position >= start_processing_position){
						processing_space_free[operand_stack.top().position] = true;
					}
				}
				operand_stack.pop();

				gdf_binary_operator operation = get_binary_operation(token);
				operators.push_back(operation);
				unary_operators.push_back(GDF_INVALID_UNARY);

				if(is_literal(left_operand) && is_literal(right_operand)){
					//both are literal have to deduce types, nuts
					//TODO: this is not working yet becuase we have to deduce the types..
					//					gdf_scalar left = get_scalar_from_string(left_operand,inputs.get_column(right_index).dtype());
					//					left_scalars.push_back(left);
					//					gdf_scalar right = get_scalar_from_string(right_operand,inputs.get_column(right_index).dtype());
					//					right_scalars.push_back(left);

					left_inputs.push_back(SCALAR_INDEX); //
				}else if(is_literal(left_operand)){
					size_t right_index = get_index(right_operand);
					// TODO: remove get_type_from_string dirty fix
					// gdf_scalar right = get_scalar_from_string(left_operand,inputs.get_column(get_index(right_operand)).dtype());
					gdf_scalar left = get_scalar_from_string(left_operand,get_type_from_string(left_operand));
					left_scalars.push_back(left);
					right_scalars.push_back(dummy_scalar);

					left_inputs.push_back(left.is_valid ? SCALAR_INDEX : SCALAR_NULL_INDEX);
					right_inputs.push_back(right_index);
				}else if(is_literal(right_operand) && !is_string(right_operand)){
					size_t left_index = get_index(left_operand);
					// TODO: remove get_type_from_string dirty fix
					// gdf_scalar right = get_scalar_from_string(right_operand,inputs.get_column(get_index(left_operand)).dtype());
					gdf_scalar right = get_scalar_from_string(right_operand,get_type_from_string(right_operand));
					right_scalars.push_back(right);
					left_scalars.push_back(dummy_scalar);

					right_inputs.push_back(right.is_valid ? SCALAR_INDEX : SCALAR_NULL_INDEX);
					left_inputs.push_back(left_index);
				}else if(is_literal(right_operand) && is_string(right_operand)){
					right_operand = right_operand.substr(1,right_operand.size()-2);
					size_t left_index = get_index(left_operand);
					gdf_column* left_column = input_columns[left_index];

					int found = left_column->dtype_info.category->get_value(right_operand.c_str());

					if(found != -1){
						gdf_data data;
						data.si32 = found;
						gdf_scalar right = {data, GDF_INT32, true};

						right_scalars.push_back(right);
						left_scalars.push_back(dummy_scalar);

						right_inputs.push_back(right.is_valid ? SCALAR_INDEX : SCALAR_NULL_INDEX);
						left_inputs.push_back(left_index);
					}
					else{ //insertar nuevo value, reemplazar columna left

						const char* str = right_operand.c_str();
						const char** strs = &str;
						NVStrings* temp_string = NVStrings::create_from_array(strs, 1);
						NVCategory* new_category = left_column->dtype_info.category->add_strings(*temp_string);
						left_column->dtype_info.category = new_category;

						size_t size_to_copy = sizeof(int32_t) * left_column->size;

						cudaMemcpyAsync(left_column->data,
							left_column->dtype_info.category->values_cptr(),
							size_to_copy,
							cudaMemcpyDeviceToDevice);
						
						int found = left_column->dtype_info.category->get_value(right_operand.c_str());

						gdf_data data;
						data.si32 = found;
						gdf_scalar right = {data, GDF_INT32, true};

						right_scalars.push_back(right);
						left_scalars.push_back(dummy_scalar);

						right_inputs.push_back(right.is_valid ? SCALAR_INDEX : SCALAR_NULL_INDEX);
						left_inputs.push_back(left_index);
					}
				}else{
					size_t left_index = get_index(left_operand);
					size_t right_index = get_index(right_operand);

					if(input_columns.size() > left_index && input_columns.size() > right_index){
						gdf_column* left_column = input_columns[left_index];
						gdf_column* right_column = input_columns[right_index];

						if(left_column->dtype == GDF_STRING_CATEGORY && right_column->dtype == GDF_STRING_CATEGORY) {
							gdf_column * process_columns[2] = {left_column, right_column};
							gdf_column * output_columns[2] = {left_column, right_column};

							//CUDF_CALL( combine_column_categories(process_columns, output_columns, 2) );
							CUDF_CALL( sync_column_categories(process_columns, output_columns, 2) );

							input_columns[left_index] = output_columns[0];
							input_columns[right_index] = output_columns[1];
						}
					}

					left_inputs.push_back(left_index);
					right_inputs.push_back(right_index);

					left_scalars.push_back(dummy_scalar);
					right_scalars.push_back(dummy_scalar);
				}
			}else if(is_unary_operator_token(token)){
				std::string left_operand = operand_stack.top().token;
				if(!is_literal(left_operand)){
					if(operand_stack.top().position >= start_processing_position){
						processing_space_free[operand_stack.top().position] = true;
					}
				}
				operand_stack.pop();

				gdf_unary_operator operation = get_unary_operation(token);
				operators.push_back(GDF_INVALID_BINARY);
				unary_operators.push_back(operation);

				if(is_literal(left_operand)){

				}else{
					size_t left_index = get_index(left_operand);
					left_inputs.push_back(left_index);
					right_inputs.push_back(-1);

					left_scalars.push_back(dummy_scalar);
					right_scalars.push_back(dummy_scalar);
				}

			} else if (is_other_binary_operator_token(token)){
				//well we can figure this out later
			}else{
				//uh oh
			}

			if(position == 0){
				//write to final output
				outputs.push_back(expression_position + num_inputs);
			}else{
				//write to temp output
				column_index_type output_position = get_first_open_position(processing_space_free,start_processing_position);
				outputs.push_back(output_position);
				//push back onto stack
				operand_stack.push({std::string("$") + std::to_string(output_position),output_position});
			}
		}else{
			if(is_literal(token) || is_string(token)){
				operand_stack.push({token,SCALAR_INDEX});
			}else{
				operand_stack.push({std::string("$" + std::to_string(new_input_indices[get_index(token)])),new_input_indices[get_index(token)]});
			}
		}
	}
}


// processing in reverse we never need to have more than TWO spaces to work in
void evaluate_expression(
		blazing_frame inputs,
		std::string expression,
		gdf_column_cpp output){

	// make temp a column of size 8 bytes so it can accomodate the largest possible size
	static CodeTimer timer;
	timer.reset();

	std::string clean_expression = clean_calcite_expression(expression);
	int position = clean_expression.size();
	std::stack<std::string> operand_stack;

	std::vector<column_index_type> final_output_positions(1);
	std::vector<gdf_column *> output_columns(1);
	output_columns[0] = output.get_gdf_column();
	std::vector<gdf_column *> input_columns;

	std::vector<gdf_dtype> output_type_expressions(1); //contains output types for columns that are expressions, if they are not expressions we skip over it
	output_type_expressions[0] = output.dtype();

	std::vector<bool> input_used_in_expression(inputs.get_size_columns(),false);
	while(position > 0){
		std::string token = get_last_token(clean_expression,&position);

		if(!is_operator_token(token) && !is_literal(token) && !is_string(token)){
			size_t index = get_index(token);
			input_used_in_expression[index] = true;
		}
	}

	std::vector<column_index_type>  left_inputs;
	std::vector<column_index_type>  right_inputs;
	std::vector<column_index_type>  outputs;

	std::vector<gdf_binary_operator>  operators;
	std::vector<gdf_unary_operator>  unary_operators;


	std::vector<gdf_scalar>  left_scalars;
	std::vector<gdf_scalar>  right_scalars;

	std::vector<column_index_type> new_column_indices(input_used_in_expression.size());
	size_t input_columns_used = 0;
	for(int i = 0; i < input_used_in_expression.size(); i++){
		if(input_used_in_expression[i]){
			new_column_indices[i] = input_columns_used;
			input_columns.push_back( inputs.get_column(i).get_gdf_column());
			input_columns_used++;

		}else{
			new_column_indices[i] = -1; //won't be uesd anyway
		}
	}

	final_output_positions[0] = input_columns_used;


	add_expression_to_plan(	inputs,
						input_columns,
						expression,
						0,
						1,
						input_columns_used,
						left_inputs,
						right_inputs,
						outputs,
						operators,
						unary_operators,
						left_scalars,
						right_scalars,
						new_column_indices);



	perform_operation( output_columns,
				input_columns,
				left_inputs,
				right_inputs,
				outputs,
				final_output_positions,
				operators,
				unary_operators,
				left_scalars,
				right_scalars,
				new_column_indices);


	// output.update_null_count();

	// Library::Logging::Logger().logInfo("-> evaluate_expression took " + std::to_string(timer.getDuration()) + " ms processing expression:\n" + expression);
}

