/*
 * CalciteExpressionParsing.h
 *
 *  Created on: Aug 7, 2018
 *      Author: felipe
 */

#ifndef CALCITEEXPRESSIONPARSING_H_
#define CALCITEEXPRESSIONPARSING_H_

#include <string>
#include <gdf/gdf.h>

class blazing_frame;

gdf_error get_operation(
		std::string operator_string,
		gdf_binary_operator * operation
);

bool is_literal(std::string operand);

bool is_digits(const std::string &str);

bool is_operator_token(std::string operand);

size_t get_index(std::string operand_string);

std::string clean_calcite_expression(std::string expression);

std::string get_last_token(std::string expression, int * position);

size_t get_index(std::string operand_string);

gdf_error get_aggregation_operation(std::string operator_string, gdf_agg_op * operation);

std::string get_string_between_outer_parentheses(std::string operator_string);

gdf_error get_output_type_expression(blazing_frame * input, gdf_dtype * output_type, gdf_dtype * max_temp_type, std::string expression);
gdf_dtype get_aggregation_output_type(gdf_dtype input_type,  gdf_agg_op aggregation);

gdf_scalar get_scalar_from_string(std::string scalar_string, gdf_dtype type);

size_t get_width_dtype(gdf_dtype type);

#endif /* CALCITEEXPRESSIONPARSING_H_ */
