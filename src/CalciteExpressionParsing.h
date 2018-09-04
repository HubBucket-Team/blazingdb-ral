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

#endif /* CALCITEEXPRESSIONPARSING_H_ */
