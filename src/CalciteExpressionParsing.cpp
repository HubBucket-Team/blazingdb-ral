#include <stack>
#include <sstream>
#include <iomanip>
#include <regex>

#include "CalciteExpressionParsing.h"
#include "DataFrame.h"
#include "StringUtil.h"

bool is_type_signed(gdf_dtype type){
	return (GDF_INT8 == type ||
			GDF_INT16 == type ||
			GDF_INT32 == type ||
			GDF_INT64 == type ||
			GDF_FLOAT32 == type ||
			GDF_FLOAT64 == type ||
			GDF_DATE32 == type ||
			GDF_DATE64 == type ||
			GDF_TIMESTAMP == type);
}

bool is_type_unsigned_numeric(gdf_dtype type){
	return (GDF_UINT8 == type ||
			GDF_UINT16 == type ||
			GDF_UINT32 == type ||
			GDF_UINT64 == type);
}

bool is_type_float(gdf_dtype type){
	return (GDF_FLOAT32 == type ||
			GDF_FLOAT64 == type);
}

bool is_numeric_type(gdf_dtype type){
	return is_type_signed(type) || is_type_unsigned_numeric(type);
}

gdf_dtype get_next_biggest_type(gdf_dtype type){
	if(type == GDF_INT8){
		return GDF_INT16;
	}else if(type == GDF_INT16){
		return GDF_INT32;
	}else if(type == GDF_INT32){
		return GDF_INT64;
	}else if(type == GDF_UINT8){
		return GDF_UINT16;
	}else if(type == GDF_UINT16){
		return GDF_UINT32;
	}else if(type == GDF_UINT32){
		return GDF_UINT64;
	}else if(type == GDF_FLOAT32){
		return GDF_FLOAT64;
	}else{
		return type;
	}
}

gdf_dtype get_aggregation_output_type(gdf_dtype input_type,  gdf_agg_op aggregation){
	if(aggregation == GDF_COUNT){
		return GDF_UINT64;
	}else if(aggregation == GDF_SUM){
		//we can assume it is numeric based on the oepration
		//here we are in an interseting situation
		//it can grow larger than the input type, to be safe we should enlarge to the greatest signed or unsigned representation
		if(is_type_float(input_type)){
			return input_type;
		}

		if(is_type_signed(input_type)){
			return GDF_INT64;
		}else{
			return GDF_UINT64;
		}
	}else if(aggregation == GDF_MIN){
		return input_type;
	}else if(aggregation == GDF_MAX){
		return input_type;
	}else if(aggregation == GDF_AVG){
		return input_type;
	}else if(aggregation == GDF_COUNT){
		return GDF_UINT64;
	}else if(aggregation == GDF_COUNT_DISTINCT){
		return GDF_UINT64;
	}else{
		return GDF_invalid;
	}

}

size_t get_width_dtype(gdf_dtype type){
	if(type == GDF_INT8){
		return 1;
	}else if(type == GDF_INT16){
		return 2;
	}else if(type == GDF_INT32){
		return 4;
	}else if(type == GDF_INT64){
		return 8;
	}else if(type == GDF_UINT8){
		return 1;
	}else if(type == GDF_UINT16){
		return 2;
	}else if(type == GDF_UINT32){
		return 4;
	}else if(type == GDF_UINT64){
		return 8;
	}else if(type == GDF_FLOAT32){
		return 4;
	}else if(type == GDF_FLOAT64){
		return 8;
	}else if(type == GDF_DATE32){
		return 4;
	}else if(type == GDF_DATE64){
		return 8;
	}else if(type == GDF_TIMESTAMP){
		return 8;
	}else if(type == GDF_CATEGORY){
		return 0;
	}else if(type == GDF_STRING){
		return 0;
	}
}

bool is_exponential_operator(gdf_binary_operator operation){
	return operation == GDF_POW;
}

bool is_arithmetic_operation(gdf_binary_operator operation){
	return (operation == GDF_ADD ||
			operation == GDF_SUB ||
			operation == GDF_MUL||
			operation == GDF_DIV ||
			operation == GDF_MOD
	);
}

bool is_comparison_operation(gdf_binary_operator operation){
	return (operation == GDF_EQUAL ||
			operation == GDF_NOT_EQUAL ||
			operation == GDF_GREATER||
			operation == GDF_GREATER_EQUAL ||
			operation == GDF_LESS ||
			operation == GDF_LESS_EQUAL
	);
}

gdf_dtype get_signed_type_from_unsigned(gdf_dtype type){
	if(type == GDF_UINT8){
		return GDF_INT16;
	}else if(type == GDF_UINT16){
		return GDF_INT32;
	}else if(type == GDF_UINT32){
		return GDF_INT64;
	}else if(type == GDF_UINT64){
		return GDF_INT64;
	}else{
		return GDF_INT64;
	}
}

gdf_dtype get_output_type(gdf_dtype input_left_type, gdf_dtype input_right_type, gdf_binary_operator operation){

	//we are only considering binary ops between numbers for now
	if(!is_numeric_type(input_left_type) || !is_numeric_type(input_right_type)){
		return GDF_invalid;
	}

	if(is_arithmetic_operation(operation)){


		if(is_type_float(input_left_type) || is_type_float(input_right_type) ){
			//the output shoudl be ther largest float type
			if(is_type_float(input_left_type) && is_type_float(input_right_type) ){
				return (get_width_dtype(input_left_type) >= get_width_dtype(input_right_type)) ? input_left_type : input_right_type;
			}else if(is_type_float(input_left_type)){
				return input_left_type;
			}else{
				return input_right_type;
			}
		}

		//ok so now we know we have now floating points left
		//so only things to worry about now are
		//if both are signed or unsigned, use largest type

		if((is_type_signed(input_left_type) && is_type_signed(input_right_type)) || (!is_type_signed(input_left_type) && !is_type_signed(input_right_type)) ){
			return (get_width_dtype(input_left_type) >= get_width_dtype(input_right_type)) ? input_left_type : input_right_type;
		}

		//now we know one is signed and the other isnt signed, if signed is larger we can just use signed version, if unsigned is larger we have to use the signed version one step up
		//e.g. an unsigned int32 requires and int64 to represent all its numbers, unsigned int64 we are just screwed :)
		if(is_type_signed(input_left_type)){
			//left signed
			//right unsigned
			if(get_width_dtype(input_left_type) > get_width_dtype(input_right_type)){
				//great the left can represent the right
				return input_left_type;
			}else{
				//right type cannot be represented by left so we need to get a signed type big enough to represent the unsigned right
				return get_signed_type_from_unsigned(input_right_type);
			}
		}else{
			//right signed
			//left unsigned
			if(get_width_dtype(input_left_type) < get_width_dtype(input_right_type)){

				return input_right_type;
			}else{

				return get_signed_type_from_unsigned(input_left_type);
			}

		}

		//convert to largest type
		//if signed and unsigned convert to signed, upgrade unsigned if possible to determine size requirements
	}else if(is_comparison_operation(operation)){
		return GDF_INT8;
	}else if(is_exponential_operator(operation)){
		//assume biggest type unsigned if left is unsigned, signed if left is signed

		if(is_type_float(input_left_type) || is_type_float(input_right_type) ){
			return GDF_FLOAT64;
		}else if(is_type_signed(input_left_type)){
			return GDF_INT64;
		}else{
			return GDF_UINT64;
		}
	}else{
		return GDF_invalid;
	}
}

bool is_date(const std::string &str){

	static const std::regex re{R"([12]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]))"};
	return std::regex_match(str, re);
}

//Todo: unit tests
int32_t get_date_32_from_string(std::string scalar_string){
	std::tm t = {};
	std::istringstream ss(scalar_string);

	if (ss >> std::get_time(&t, "%Y-%m-%d")){
		int32_t tr = std::mktime(&t);
		int32_t seconds_in_a_day = 60*60*24;
		return tr / seconds_in_a_day;
	}
	else{
		throw std::invalid_argument("Invalid datetime format");
	}
}

int64_t get_date_64_from_string(std::string scalar_string){

	std::tm t = {};
	std::istringstream ss(scalar_string);

	if (ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S")){
		int64_t tr = std::mktime(&t);
		return tr;
	}
	else{
		throw std::invalid_argument("Invalid datetime format");
	}
}

//Todo: Consider cases with different unit: ms, us, or ns
int64_t get_timestamp_from_string(std::string scalar_string){

	std::tm t = {};
	std::istringstream ss(scalar_string);

	if (ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S")){
		int64_t tr = std::mktime(&t);
		return tr;
	}
	else{
		throw std::invalid_argument("Invalid timestamp format");
	}
}

gdf_scalar get_scalar_from_string(std::string scalar_string, gdf_dtype type){
	/*
	 * void*    invd;
int8_t   si08;
int16_t  si16;
int32_t  si32;
int64_t  si64;
uint8_t  ui08;
uint16_t ui16;
uint32_t ui32;
uint64_t ui64;
float    fp32;
double   fp64;
int32_t  dt32;  // GDF_DATE32
int64_t  dt64;  // GDF_DATE64
int64_t  tmst;  // GDF_TIMESTAMP
};*/
	if(type == GDF_INT8){
		gdf_data data;
		data.si08 = stoi(scalar_string);
		return {data, GDF_INT8, true};

	}else if(type == GDF_INT16){
		gdf_data data;
		data.si16 = stoi(scalar_string);
		return {data, GDF_INT16, true};
	}else if(type == GDF_INT32){
		gdf_data data;
		data.si32 = stoi(scalar_string);
		return {data, GDF_INT32, true};
	}else if(type == GDF_INT64){
		gdf_data data;
		data.si64 = stoll(scalar_string);
		return {data, GDF_INT64, true};
	}else if(type == GDF_UINT8){
		gdf_data data;
		data.ui08 = stoull(scalar_string);
		return {data, GDF_UINT8, true};
	}else if(type == GDF_UINT16){
		gdf_data data;
		data.ui16 = stoull(scalar_string);
		return {data, GDF_UINT16, true};
	}else if(type == GDF_UINT32){
		gdf_data data;
		data.ui32 = stoull(scalar_string);
		return {data, GDF_UINT32, true};
	}else if(type == GDF_UINT64){
		gdf_data data;
		data.ui64 = stoull(scalar_string);
		return {data, GDF_UINT64, true};
	}else if(type == GDF_FLOAT32){
		gdf_data data;
		data.fp32 = stof(scalar_string);
		return {data, GDF_FLOAT32, true};
	}else if(type == GDF_FLOAT64){
		gdf_data data;
		data.fp64 = stod(scalar_string);
		return {data, GDF_FLOAT64, true};
	}else if(type == GDF_DATE32){
		//TODO: convert date literals!!!!
		gdf_data data;
		//string format o
		data.dt32 = get_date_32_from_string(scalar_string);
		return {data, GDF_DATE32, true};
	}else if(type == GDF_DATE64){
		gdf_data data;
		data.dt64 = get_date_64_from_string(scalar_string);
		return {data, GDF_DATE64, true};
	}else if(type == GDF_TIMESTAMP){
		//Todo: specific the unit
		gdf_data data;
		data.tmst = get_timestamp_from_string(scalar_string);
		return {data, GDF_TIMESTAMP, true};
	}
}

//must pass in temp type as invalid if you are not setting it to something to begin with
gdf_error get_output_type_expression(blazing_frame * input, gdf_dtype * output_type, gdf_dtype * max_temp_type, std::string expression){
	std::string clean_expression = clean_calcite_expression(expression);
	int position = clean_expression.size();
	if(*max_temp_type == GDF_invalid){
		*max_temp_type = GDF_INT8;
	}

	std::stack<gdf_dtype> operands;
	while(position > 0){
		std::string token = get_last_token(clean_expression,&position);
		//std::cout<<"Token is ==> "<<token<<"\n";

		if(is_operator_token(token)){

			gdf_dtype left_operand = operands.top();
				operands.pop();
				gdf_dtype right_operand = operands.top();
				operands.pop();

				if(left_operand == GDF_invalid){


					if(right_operand == GDF_invalid){
						return GDF_INVALID_API_CALL;
					}else{

						left_operand = right_operand;

					}
				}else{
					if(right_operand == GDF_invalid){
						right_operand = left_operand;
					}
				}
				gdf_binary_operator operation;
				gdf_error err = get_operation(token,&operation);
				operands.push(get_output_type(left_operand,right_operand,operation));
				if(position > 0 && get_width_dtype(operands.top()) > get_width_dtype(*max_temp_type)){
					*max_temp_type = operands.top();
				}
		}else{
			if(is_literal(token)){
				operands.push(GDF_invalid);
			}else{
				operands.push(input->get_column(get_index(token)).dtype() );
			}

		}
	}
	*output_type = operands.top();
	return GDF_SUCCESS;
}


gdf_error get_aggregation_operation(std::string operator_string, gdf_agg_op * operation){

	operator_string = operator_string.substr(
			operator_string.find("=[") + 2,
			(operator_string.find("]") - (operator_string.find("=[") + 2))
			);
	operator_string = StringUtil::replace(operator_string,"COUNT(DISTINCT","COUNT_DISTINCT");
	//remove expression
	operator_string = operator_string.substr(0,operator_string.find("("));
	if(operator_string == "SUM"){
		*operation = GDF_SUM;
	}else if(operator_string == "AVG"){
		*operation = GDF_AVG;
	}else if(operator_string == "MIN"){
		*operation = GDF_MIN;
	}else if(operator_string == "MAX"){
		*operation = GDF_MAX;
	}else if(operator_string == "COUNT"){
		*operation = GDF_COUNT;
	}else if(operator_string == "COUNT_DISTINCT"){
		*operation = GDF_COUNT_DISTINCT;
	}else{
		return GDF_UNSUPPORTED_DTYPE;
	}
	return GDF_SUCCESS;
}


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
	}else if(operator_string == "+"){
		*operation = GDF_ADD;
	}else if(operator_string == "-"){
		*operation = GDF_SUB;
	}else if(operator_string == "*"){
		*operation = GDF_MUL;
	}else if(operator_string == "/"){
		*operation = GDF_DIV;
	}else if(operator_string == "POWER"){
		*operation = GDF_POW;
	}else if(operator_string == "MOD"){
		*operation = GDF_MOD;
	}else if(operator_string == "AND"){
		*operation = GDF_MUL;
	}
	else{
		return GDF_UNSUPPORTED_DTYPE;
	}
	return GDF_SUCCESS;
}

bool is_literal(std::string operand){
	return operand[0] != '$';
}

bool is_digits(const std::string &str)
{
	return str.find_first_not_of("0123456789.") == std::string::npos;
}

bool is_integer(const std::string &s) {
  static const std::regex re{"-?\\d+"};
  return std::regex_match(s, re);
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

bool is_operator_token(std::string operand) {
  return (operand[0] != '$' && !is_digits(operand) && !is_date(operand)
          && !is_integer(operand));
}

size_t get_index(std::string operand_string){
	size_t start = 1;
	return std::stoull (operand_string.substr(1,operand_string.size()-1),0);
}

std::string aggregator_to_string(gdf_agg_op aggregation){
	if(aggregation == GDF_COUNT){
			return "count";
		}else if(aggregation == GDF_SUM){
			return "sum";
		}else if(aggregation == GDF_MIN){
			return "min";
		}else if(aggregation == GDF_MAX){
			return "max";
		}else if(aggregation == GDF_AVG){
			return "avg";
		}else if(aggregation == GDF_COUNT_DISTINCT){
			return "count_distinct";
		}else{
			return "";
		}
}
std::string clean_calcite_expression(std::string expression){
	//TODO: this is very hacky, the proper way is to remove this in calcite
	StringUtil::findAndReplaceAll(expression," NOT NULL","");
	StringUtil::findAndReplaceAll(expression,"):DOUBLE","");
	StringUtil::findAndReplaceAll(expression,"CAST(","");


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

std::string get_string_between_outer_parentheses(std::string input_string){
	int start_pos, end_pos;
	start_pos = input_string.find("(");
	end_pos = input_string.find(")");
	if(start_pos == input_string.npos || end_pos == input_string.npos || end_pos < start_pos){
		return "";
	}
	start_pos++;
	//end_pos--;

	return input_string.substr(start_pos,end_pos - start_pos);
}
