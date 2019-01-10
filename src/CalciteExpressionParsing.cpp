#include <stack>
#include <sstream>
#include <iomanip>
#include <regex>

#include <blazingdb/io/Util/StringUtil.h>

#include "CalciteExpressionParsing.h"
#include "DataFrame.h"


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

bool is_type_float(gdf_dtype type){
	return (GDF_FLOAT32 == type ||
			GDF_FLOAT64 == type);
}

bool is_date_type(gdf_dtype type){
	return (GDF_DATE32 == type ||
			GDF_DATE64 == type ||
			GDF_TIMESTAMP == type);
}

//TODO percy noboa see upgrade to uints
//bool is_type_unsigned_numeric(gdf_dtype type){
//	return (GDF_UINT8 == type ||
//			GDF_UINT16 == type ||
//			GDF_UINT32 == type ||
//			GDF_UINT64 == type);
//}

//TODO percy noboa see upgrade to uints
bool is_numeric_type(gdf_dtype type){
	//return is_type_signed(type) || is_type_unsigned_numeric(type);
	return is_type_signed(type);
}

gdf_dtype get_next_biggest_type(gdf_dtype type){
	//	if(type == GDF_INT8){
	//		return GDF_INT16;
	//	}else if(type == GDF_INT16){
	//		return GDF_INT32;
	//	}else if(type == GDF_INT32){
	//		return GDF_INT64;
	//	}else if(type == GDF_UINT8){
	//		return GDF_UINT16;
	//	}else if(type == GDF_UINT16){
	//		return GDF_UINT32;
	//	}else if(type == GDF_UINT32){
	//		return GDF_UINT64;
	//	}else if(type == GDF_FLOAT32){
	//		return GDF_FLOAT64;
	//	}else{
	//		return type;
	//	}
	//TODO felipe percy noboa see upgrade to uints
	if(type == GDF_INT8){
		return GDF_INT16;
	}else if(type == GDF_INT16){
		return GDF_INT32;
	}else if(type == GDF_INT32){
		return GDF_INT64;
	}else if(type == GDF_FLOAT32){
		return GDF_FLOAT64;
	}else{
		return type;
	}
}


// TODO all these return types need to be revisited later. Right now we have issues with some aggregators that only support returning the same input type. Also pygdf does not currently support unsigned types (for example count should return and unsigned type)
gdf_dtype get_aggregation_output_type(gdf_dtype input_type,  gdf_agg_op aggregation, std::size_t group_size){
	if(aggregation == GDF_COUNT){
		//return GDF_UINT64;
		//TODO felipe percy noboa see upgrade to uints
		return GDF_INT64;
	}else if(aggregation == GDF_SUM){
		return input_type;
		if (group_size == 0) {
			return input_type;
		}

		//we can assume it is numeric based on the oepration
		//here we are in an interseting situation
		//it can grow larger than the input type, to be safe we should enlarge to the greatest signed or unsigned representation
		if(is_type_float(input_type)){
			return input_type;
		}

		//TODO felipe percy noboa see upgrade to uints
		//if(is_type_signed(input_type)){
		return GDF_INT64;
		//}
		//else{
		//	return GDF_UINT64;
		//}
	}else if(aggregation == GDF_MIN){
		return input_type;
	}else if(aggregation == GDF_MAX){
		return input_type;
	}else if(aggregation == GDF_AVG){
		return GDF_FLOAT64;
	}else if(aggregation == GDF_COUNT){

		//return GDF_UINT64;
		//TODO felipe percy noboa see upgrade to uints


		return GDF_INT64;
	}else if(aggregation == GDF_COUNT_DISTINCT){

		//return GDF_UINT64;
		//TODO felipe percy noboa see upgrade to uints

		return GDF_INT64;
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
		//}
		//TODO felipe percy noboa see upgrade to uints
		//	else if(type == GDF_UINT8){
		//		return 1;
		//	}else if(type == GDF_UINT16){
		//		return 2;
		//	}else if(type == GDF_UINT32){
		//		return 4;
		//	}else if(type == GDF_UINT64){
		//		return 8;
	}else if(type == GDF_FLOAT32)
	{
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
	return type;
	//TODO felipe percy noboa see upgrade to uints
	//	if(type == GDF_UINT8){
	//		return GDF_INT16;
	//	}else if(type == GDF_UINT16){
	//		return GDF_INT32;
	//	}else if(type == GDF_UINT32){
	//		return GDF_INT64;
	//	}else if(type == GDF_UINT64){
	//		return GDF_INT64;
	//	}else{
	//		return GDF_INT64;
	//	}
}

gdf_dtype get_output_type(gdf_dtype input_left_type, gdf_unary_operator operation){
	if(is_date_type(input_left_type)){
		return GDF_INT16;
	}else{
		return input_left_type;
	}
}

gdf_dtype get_output_type(gdf_dtype input_left_type, gdf_dtype input_right_type, gdf_other_binary_operator operation){
	
	// the only gdf_other_binary_operator we have right now is COALESCE where we will except both sides to be the same type
	if (input_left_type == GDF_invalid)
		return input_right_type;
	else if (input_right_type == GDF_invalid)
		return input_left_type;
	else if (input_left_type == input_right_type)
		return input_right_type;
	else 
		return GDF_invalid;
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
			//		}else if(is_type_signed(input_left_type)){
			//			return GDF_INT64;
		}else{
			//TODO felipe percy noboa see upgrade to uints
			//return GDF_UINT64;
			return GDF_INT64;
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
	}
	//	else if(type == GDF_UINT8){
	//		gdf_data data;
	//		data.ui08 = stoull(scalar_string);
	//		return {data, GDF_UINT8, true};
	//	}else if(type == GDF_UINT16){
	//		gdf_data data;
	//		data.ui16 = stoull(scalar_string);
	//		return {data, GDF_UINT16, true};
	//	}else if(type == GDF_UINT32){
	//		gdf_data data;
	//		data.ui32 = stoull(scalar_string);
	//		return {data, GDF_UINT32, true};
	//	}else if(type == GDF_UINT64){
	//		gdf_data data;
	//		data.ui64 = stoull(scalar_string);
	//		return {data, GDF_UINT64, true};
	//	}
	else if(type == GDF_FLOAT32){
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
			if(is_binary_operator_token(token) || is_other_binary_operator_token(token)){
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
				if (is_binary_operator_token(token)){
					gdf_binary_operator operation;
					gdf_error err = get_operation(token,&operation);
					operands.push(get_output_type(left_operand,right_operand,operation));
				} else {
					gdf_other_binary_operator operation;
					gdf_error err = get_operation(token,&operation);
					operands.push(get_output_type(left_operand,right_operand,operation));
				}
				if(position > 0 && get_width_dtype(operands.top()) > get_width_dtype(*max_temp_type)){
					*max_temp_type = operands.top();
				}
			}else if(is_unary_operator_token(token)){
				gdf_dtype left_operand = operands.top();
				operands.pop();

				gdf_unary_operator operation;
				gdf_error err = get_operation(token,&operation);

				operands.push(get_output_type(left_operand,operation));
				if(position > 0 && get_width_dtype(operands.top()) > get_width_dtype(*max_temp_type)){
					*max_temp_type = operands.top();
				}
			} else {
				return GDF_INVALID_API_CALL;
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
		return GDF_INVALID_API_CALL;
	}
	return GDF_SUCCESS;
}

gdf_error get_operation(
		std::string operator_string,
		gdf_unary_operator * operation
){
	if(operator_string == "NOT"){
		*operation = GDF_NOT;
	}else if(operator_string == "SIN"){
		*operation = GDF_SIN;
	}else if(operator_string == "ASIN"){
		*operation = GDF_ASIN;
	}else if(operator_string == "COS"){
		*operation = GDF_COS;
	}else if(operator_string == "ACOS"){
		*operation = GDF_ACOS;
	}else if(operator_string == "TAN"){
		*operation = GDF_TAN;
	}else if(operator_string == "ATAN"){
		*operation = GDF_ATAN;
	}else if(operator_string == "BL_FLOUR"){
		*operation = GDF_FLOOR;
	}else if(operator_string == "CEIL"){
		*operation = GDF_CEIL;
	}else if(operator_string == "ABS"){
		*operation = GDF_ABS;
	}else if(operator_string == "LOG10"){
		*operation = GDF_LOG;
	}else if(operator_string == "LN"){
		*operation = GDF_LN;
	}else if(operator_string == "BL_YEAR"){
		*operation = GDF_YEAR;
	}else if(operator_string == "BL_MONTH"){
		*operation = GDF_MONTH;
	}else if(operator_string == "BL_DAY"){
		*operation = GDF_DAY;
	}else if(operator_string == "BL_HOUR"){
		*operation = GDF_HOUR;
	}else if(operator_string == "BL_MINUTE"){
		*operation = GDF_MINUTE;
	}else if(operator_string == "BL_SECOND"){
		*operation = GDF_SECOND;
	}else {
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
	}else if(operator_string == "OR"){
		*operation = GDF_ADD;
	}else{
		return GDF_INVALID_API_CALL;
	}
	return GDF_SUCCESS;
}

gdf_error get_operation(
		std::string operator_string,
		gdf_other_binary_operator * operation
){
	if(operator_string == "COALESCE"){
		*operation = GDF_COALESCE;
	}else{
		return GDF_INVALID_API_CALL;
	}
	return GDF_SUCCESS;
}

bool is_binary_operator_token(std::string token){
	gdf_binary_operator op;
	return get_operation(token,&op) == GDF_SUCCESS;
}

bool is_unary_operator_token(std::string token){
	gdf_unary_operator op;
	return get_operation(token,&op) == GDF_SUCCESS;
}

bool is_other_binary_operator_token(std::string token){
	gdf_other_binary_operator op;
	return get_operation(token,&op) == GDF_SUCCESS;
}

bool is_literal(std::string operand){
	return operand[0] != '$';
}

bool is_number(const std::string &s) {
	static const std::regex re{R""(^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$)""};
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
	return (operand[0] != '$' && !is_number(operand) && !is_date(operand));
}

size_t get_index(std::string operand_string){
	std::string cleaned_expression = clean_calcite_expression(operand_string);
	if (cleaned_expression.length() == 0) {
		return 0;
	}
	size_t start = 1;
	return std::stoull (cleaned_expression.substr(1,cleaned_expression.size()-1),0);
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

//interprets the expression and if is n-ary and logical, then returns their corresponding binary version
std::string expand_if_logical_op(std::string expression){


	std::string output = expression;
	int start_pos = 0;

	while(start_pos < expression.size()){

		std::vector<bool> is_quoted_vector = StringUtil::generateQuotedVector(expression);

		int first_and = StringUtil::findFirstNotInQuotes(expression, "AND(", start_pos, is_quoted_vector); // returns -1 if not found
		int first_or = StringUtil::findFirstNotInQuotes(expression, "OR(", start_pos, is_quoted_vector); // returns -1 if not found

		int first = -1;
		std::string op = "";
		if (first_and >= 0) {
			if (first_or >= 0 && first_or < first_and){
				first = first_or;
				op = "OR(";
			} else {
				first = first_and;
				op = "AND(";
			}
		} else {
			first = first_or;
			op = "OR(";
		}

		if (first >= 0) {
			int expression_start = first + op.size() - 1;
			int expression_end = find_closing_char(expression, expression_start);

			std::string rest = expression.substr(expression_start+1, expression_end-(expression_start+1));
			// the trim flag is false because trimming the expressions cause malformmed ones
			std::vector<std::string> processed = get_expressions_from_expression_list(rest, false);

			if(processed.size() == 2){ //is already binary
				start_pos = expression_start;
				continue;
			} else {
				start_pos = first;
			}

			output = expression.substr(0, first);
			for(size_t I=0; I<processed.size()-1; I++){
				output += op;
				start_pos += op.size();
			}

			output += processed[0] + ",";
			for(size_t I=1; I<processed.size()-1; I++){
				output += processed[I] + "),";
			}
			output += processed[processed.size()-1] + ")";

			if (expression_end < expression.size() - 1){
				output += expression.substr(expression_end + 1);
			}
			expression = output;
		} else {
			return output;
		}
	}

	return output;
}

// Different of clean_calcite_expression, this function only remove casting tokens
std::string clean_project_expression(std::string expression){
	//TODO: this is very hacky, the proper way is to remove this in calcite
	StringUtil::findAndReplaceAll(expression,"CAST(","");
	StringUtil::findAndReplaceAll(expression,"):BIGINT","");

	return expression;
}

std::string clean_calcite_expression(std::string expression){
	//TODO: this is very hacky, the proper way is to remove this in calcite
	// std::cout << ">>>>>>>>> " << expression << std::endl;
	static const std::regex re{R""(CASE\(IS NOT NULL\((\W\(.+?\)|.+)\), \1, (\W\(.+?\)|.+)\))"", std::regex_constants::icase};
	expression = std::regex_replace(expression, re, "COALESCE($1, $2)");
	// std::cout << "+++++++++ " << expression << std::endl;

	StringUtil::findAndReplaceAll(expression," NOT NULL","");
	StringUtil::findAndReplaceAll(expression,"):DOUBLE","");
	StringUtil::findAndReplaceAll(expression,"CAST(","");
	StringUtil::findAndReplaceAll(expression,"EXTRACT(FLAG(YEAR), ","BL_YEAR(");
	StringUtil::findAndReplaceAll(expression,"EXTRACT(FLAG(MONTH), ","BL_MONTH(");
	StringUtil::findAndReplaceAll(expression,"EXTRACT(FLAG(DAY), ","BL_DAY(");
	StringUtil::findAndReplaceAll(expression,"FLOOR(","BL_FLOUR(");


// we want this "CASE(IS NOT NULL($1), $1, -1)" to become this: "COALESCE($1, -1)"
// "CASE(IS NOT NULL((-($1, $2))), -($1, $2), -1)" to become this: "COALESCE(-($1, $2), -1)"
// "+(CASE(IS NOT NULL((-($1, $2))), -($1, $2), -1), *($4, $5))" to become this: "+(COALESCE(-($1, $2), -1), *($4, $5)) "

	// std::string coalesce_identifier = "CASE(IS NOT NULL(";
	// size_t pos = expression.find(coalesce_identifier);
	// int endOfFirstArg = find_closing_char(expression, pos + coalesce_identifier.length() - 1) ;
	// // this should be in this example "$1"
	// std::string firstArg = expression.substring(pos + coalesce_identifier.length(), endOfFirstArg - (pos + coalesce_identifier.length()));
	// std::string 
	



	expression = expand_if_logical_op(expression);

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
	end_pos = input_string.rfind(")");
	if(start_pos == input_string.npos || end_pos == input_string.npos || end_pos < start_pos){
		return "";
	}
	start_pos++;
	//end_pos--;

	return input_string.substr(start_pos,end_pos - start_pos);
}

int find_closing_char(const std::string & expression, int start) {

	char openChar = expression[start];

	char closeChar = openChar;
	if (openChar == '('){
		closeChar = ')';
	} else if (openChar == '['){
		closeChar = ']';
	} else {
		// TODO throw error
		return -1;
	}

	int curInd = start + 1;
	int closePos = curInd;
	int depth = 1;
	bool inQuotes = false;

	while (curInd < expression.size()){
		if (inQuotes){
			if (expression[curInd] == '\''){
				if (!(curInd + 1 < expression.size() && expression[curInd + 1] == '\'')){ // if we are in quotes and we get a double single quotes, that is an escaped quotes
					inQuotes = false;
				}
			}
		} else {
			if (expression[curInd] == '\''){
				inQuotes = true;
			} else if (expression[curInd] == openChar){
				depth++;
			} else if (expression[curInd] == closeChar){
				depth--;
				if (depth == 0){
					return curInd;
				}
			}
		}
		curInd++;
	}
	// TODO throw error
	return -1;
}

// takes a comma delimited list of expressions and splits it into separate expressions
std::vector<std::string> get_expressions_from_expression_list(const std::string & combined_expression, bool trim){

	std::vector<std::string> expressions;

	int curInd = 0;
	int curStart = 0;
	bool inQuotes = false;
	int parenthesisDepth = 0;
	int sqBraketsDepth = 0;
	while (curInd < combined_expression.size()){
		if (inQuotes){
			if (combined_expression[curInd] == '\''){
				if (!(curInd + 1 < combined_expression.size() && combined_expression[curInd + 1] == '\'')){ // if we are in quotes and we get a double single quotes, that is an escaped quotes
					inQuotes = false;
				}
			}
		} else {
			if (combined_expression[curInd] == '\''){
				inQuotes = true;
			} else if (combined_expression[curInd] == '('){
				parenthesisDepth++;
			} else if (combined_expression[curInd] == ')'){
				parenthesisDepth--;
			} else if (combined_expression[curInd] == '['){
				sqBraketsDepth++;
			} else if (combined_expression[curInd] == ']'){
				sqBraketsDepth--;
			} else if (combined_expression[curInd] == ',' && parenthesisDepth == 0 && sqBraketsDepth == 0){
				std::string exp = combined_expression.substr(curStart, curInd - curStart);

				if(trim)
					expressions.push_back(StringUtil::ltrim(exp));
				else
					expressions.push_back(exp);

				curStart = curInd + 1;
			}
		}
		curInd++;
	}

	if (curStart < combined_expression.size() && curInd <= combined_expression.size()){
		std::string exp = combined_expression.substr(curStart, curInd - curStart);

		if(trim)
			expressions.push_back(StringUtil::trim(exp));
		else
			expressions.push_back(exp);
	}

	return expressions;
}
