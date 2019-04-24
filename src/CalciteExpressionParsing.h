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
#include "cudf/types.h"

class blazing_frame;

gdf_binary_operator get_binary_operation(std::string operator_string);

gdf_unary_operator get_unary_operation(std::string operator_string);

bool is_binary_operator_token(std::string token);

bool is_unary_operator_token(std::string token);



bool is_literal(std::string operand);

bool is_string(const std::string &operand);

bool is_digits(const std::string &str);

bool is_operator_token(std::string operand);

bool is_date(const std::string &str);

size_t get_index(std::string operand_string);

// interprets the expression and if is n-ary and logical, then returns their corresponding binary version
std::string expand_if_logical_op(std::string expression);

std::string clean_project_expression(std::string expression);

std::string clean_calcite_expression(std::string expression);

std::string get_last_token(std::string expression, int * position);

gdf_agg_op get_aggregation_operation(std::string operator_string);

std::string get_string_between_outer_parentheses(std::string operator_string);

gdf_dtype get_output_type_expression(blazing_frame * input, gdf_dtype * max_temp_type, std::string expression);

gdf_dtype get_aggregation_output_type(gdf_dtype input_type,  gdf_agg_op aggregation, std::size_t group_size);

gdf_dtype get_type_from_string(std::string scalar_string);
gdf_scalar get_scalar_from_string(std::string scalar_string, gdf_dtype type);

size_t get_width_dtype(gdf_dtype type);

std::string aggregator_to_string(gdf_agg_op operation);

// takes an expression and given a starting index pointing at either ( or [, it finds the corresponding closing char ) or ]
int find_closing_char(const std::string & expression, int start);

// takes a comma delimited list of expressions and splits it into separate expressions
// if the flag trim is true, leading and trailing spaces are removed
std::vector<std::string> get_expressions_from_expression_list(std::string & combined_expressions, bool trim = true);

bool is_null(std::string token);

bool is_type_signed(gdf_dtype type);

bool is_type_float(gdf_dtype type);
bool is_date_type(gdf_dtype type);
gdf_dtype get_output_type(gdf_dtype input_left_type, gdf_dtype input_right_type, gdf_binary_operator operation);
gdf_dtype get_output_type(gdf_dtype input_left_type, gdf_unary_operator operation);

std::string get_named_expression(std::string query_part, std::string expression_name);

bool contains_evaluation(std::string expression);

#endif /* CALCITEEXPRESSIONPARSING_H_ */
