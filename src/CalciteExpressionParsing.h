/*
 * CalciteExpressionParsing.h
 *
 *  Created on: Aug 7, 2018
 *      Author: felipe
 */

#ifndef CALCITEEXPRESSIONPARSING_H_
#define CALCITEEXPRESSIONPARSING_H_

#include <string>
#include <vector>

#include "gdf_wrapper/gdf_wrapper.cuh"

class blazing_frame;

gdf_error get_operation(
		std::string operator_string,
		gdf_binary_operator * operation
);

bool is_literal(std::string operand);

bool is_digits(const std::string &str);

bool is_operator_token(std::string operand);

bool is_date(const std::string &str);

size_t get_index(std::string operand_string);

// interprets the expression and if is n-ary and logical, then returns their corresponding binary version
std::string expand_if_logical_op(std::string expression);

std::string clean_calcite_expression(std::string expression);

std::string get_last_token(std::string expression, int * position);

size_t get_index(std::string operand_string);

gdf_error get_aggregation_operation(std::string operator_string, gdf_agg_op * operation);

std::string get_string_between_outer_parentheses(std::string operator_string);

gdf_error get_output_type_expression(blazing_frame * input, gdf_dtype * output_type, gdf_dtype * max_temp_type, std::string expression);
gdf_dtype get_aggregation_output_type(gdf_dtype input_type,  gdf_agg_op aggregation, std::size_t group_size);

gdf_scalar get_scalar_from_string(std::string scalar_string, gdf_dtype type);

size_t get_width_dtype(gdf_dtype type);

std::string aggregator_to_string(gdf_agg_op operation);

// takes an expression and given a starting index pointing at either ( or [, it finds the corresponding closing char ) or ]
int find_closing_char(const std::string & expression, int start);

// takes a comma delimited list of expressions and splits it into separate expressions
std::vector<std::string> get_expressions_from_expression_list(const std::string & combined_expressions);

#endif /* CALCITEEXPRESSIONPARSING_H_ */
