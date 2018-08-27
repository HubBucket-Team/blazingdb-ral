
#include "CalciteExpressionParsing.h"


gdf_error get_operation(
		std::string operator_string,
		gdf_binary_operator * operation
){
	if(operator_string == "="){
		*operation = GDF_EQUAL;
	}else if(operator_string == "<>"){
		*operation = GDF_NOT_EQUAL;
	}else if(operator_string == ">"){
		*operation = GDF_GREATER;
	}else if(operator_string == ">="){
		*operation = GDF_GREATER_EQUAL;
	}else if(operator_string == "<"){
		*operation = GDF_LESS;
	}else if(operator_string == "<="){
		*operation = GDF_LESS_EQUAL;
	}else{
		return GDF_UNSUPPORTED_DTYPE;
	}
	return GDF_SUCCESS;
}

bool is_literal(std::string operand){
	return operand[0] != '$';
}

bool is_digits(const std::string &str)
{
    return str.find_first_not_of("0123456789") == std::string::npos;
}

std::string get_last_token(std::string expression, int * position){
	size_t old_position = *position;
	*position = expression.find_last_of(' ',*position - 1);
	if(*position == expression.npos){
		//should be at the last token
		*position = 0;
		return expression.substr(0,old_position);
	}else{
		return expression.substr(*position + 1, old_position- (*position + 1));
	}
}


bool is_operator_token(std::string operand){

	return (operand[0] != '$' && ! is_digits(operand));

}

size_t get_index(std::string operand_string){
	size_t start = 1;
	return std::stoull (operand_string.substr(1,operand_string.size()-1),0);
}


std::string clean_calcite_expression(std::string expression){
	std::string new_string = "";
	new_string.reserve(expression.size());

	for(int i = 0; i < expression.size(); i++){
		if(expression[i] == '('){
			new_string.push_back(' ');

		}else if(expression[i] != ',' && expression[i] != ')'){
			new_string.push_back(expression.at(i));
		}
	}

	return new_string;
}
