#include "CalciteInterpreter.h"

#include <blazingdb/io/Library/Logging/Logger.h>
#include <blazingdb/io/Util/StringUtil.h>

#include <algorithm>
#include <thread>
#include <regex>
#include <string>
#include <set>

#include "Utils.cuh"
#include "LogicalFilter.h"
#include "ResultSetRepository.h"
#include "JoinProcessor.h"
#include "ColumnManipulation.cuh"
#include "CalciteExpressionParsing.h"
#include "CodeTimer.h"
#include "Traits/RuntimeTraits.h"
#include "cuDF/Allocator.h"
#include "Interpreter/interpreter_cpp.h"

const std::string LOGICAL_JOIN_TEXT = "LogicalJoin";
const std::string LOGICAL_UNION_TEXT = "LogicalUnion";
const std::string LOGICAL_SCAN_TEXT = "TableScan";
const std::string LOGICAL_AGGREGATE_TEXT = "LogicalAggregate";
const std::string LOGICAL_PROJECT_TEXT = "LogicalProject";
const std::string LOGICAL_SORT_TEXT = "LogicalSort";
const std::string LOGICAL_FILTER_TEXT = "LogicalFilter";
const std::string ASCENDING_ORDER_SORT_TEXT = "ASC";
const std::string DESCENDING_ORDER_SORT_TEXT = "DESC";


bool is_join(std::string query_part){
	return (query_part.find(LOGICAL_JOIN_TEXT) != std::string::npos);
}

bool is_union(std::string query_part){
	return (query_part.find(LOGICAL_UNION_TEXT) != std::string::npos);
}

bool is_project(std::string query_part){
	return (query_part.find(LOGICAL_PROJECT_TEXT) != std::string::npos);
}

bool is_aggregate(std::string query_part){
	return (query_part.find(LOGICAL_AGGREGATE_TEXT) != std::string::npos);
}

bool is_sort(std::string query_part){
	return (query_part.find(LOGICAL_SORT_TEXT) != std::string::npos);
}

bool is_scan(std::string query_part){
	return (query_part.find(LOGICAL_SCAN_TEXT) != std::string::npos);
}

bool is_filter(std::string query_part){
	return (query_part.find(LOGICAL_FILTER_TEXT) != std::string::npos);
}

int count_string_occurrence(std::string haystack, std::string needle){
	int position = haystack.find(needle, 0);
	int count = 0;
	while (position != std::string::npos)
	{
		count++;
		position = haystack.find(needle, position + needle.size());
	}

	return count;
}

bool is_double_input(std::string query_part){
	if(is_join(query_part)){
		return true;
	}else if(is_union(query_part)){
		return true;
	}else{
		return false;
	}
}

//Input: [[hr, emps]] or [[emps]] Output: hr.emps or emps
std::string extract_table_name(std::string query_part){
	size_t start = query_part.find("[[") + 2;
	size_t end = query_part.find("]]");
	std::string table_name_text = query_part.substr(start,end-start);
	std::vector<std::string> table_parts = StringUtil::split(
			table_name_text
			,',');
	std::string table_name = "";
	for(int i = 0; i < table_parts.size(); i++){
		if(table_parts[i][0] == ' '){
			table_parts[i] = table_parts[i].substr(1,table_parts[i].size()-1);
		}
		table_name += table_parts[i];
		if(i != table_parts.size() -1 ){
			table_name += ".";
		}
	}

	return table_name;

}

/*void create_output_and_evaluate(size_t size, gdf_column * output, gdf_column * temp){
	int width;
	get_column_byte_width(output, &width);
	create_gdf_column(output,output->dtype,size,nullptr,width);
	create_gdf_column(temp,GDF_INT64,size,nullptr,8);


	gdf_error err = evaluate_expression(
			input,
			get_expression(query_part),
			&output,
			&temp);

}*/

std::string get_condition_expression(std::string query_part){
	return get_named_expression(query_part,"condition");
}

bool contains_evaluation(std::string expression){
	std::string cleaned_expression = clean_project_expression(expression);
	return (cleaned_expression.find("(") != std::string::npos);
}

gdf_error create_null_value_gdf_column(int64_t output_value,
		gdf_dtype output_type,
		std::size_t output_size,
		std::string&& output_name,
		gdf_column_cpp& output_column,
		std::vector<gdf_column_cpp>& output_vector) {
	output_column.create_gdf_column(output_type,
			output_size,
			&output_value,
			get_width_dtype(output_type),
			output_name);

	int invalid = 0;
	CheckCudaErrors(cudaMemcpy(output_column.valid(), &invalid, 1, cudaMemcpyHostToDevice));

	output_vector.pop_back();
	output_vector.emplace_back(output_column);

	return GDF_SUCCESS;
}

gdf_error perform_avg(gdf_column* column_output, gdf_column* column_input) {
	gdf_error error;
	gdf_column_cpp column_avg;
	uint64_t avg_sum = 0;
	uint64_t avg_count = column_input->size;
	{
		auto dtype = column_input->dtype;
		auto dtype_size = get_width_dtype(dtype);
		column_avg.create_gdf_column(dtype, 1, nullptr, dtype_size);
		error = gdf_sum(column_input, column_avg.get_gdf_column()->data, dtype_size);
		if (error != GDF_SUCCESS) {
			return error;
		}
		CheckCudaErrors(cudaMemcpy(&avg_sum, column_avg.get_gdf_column()->data, dtype_size, cudaMemcpyDeviceToHost));
	}
	{
		auto dtype = column_output->dtype;
		auto dtype_size = get_width_dtype(dtype);
		if (Ral::Traits::is_dtype_float32(dtype)) {
			float result = (float) avg_sum / (float) avg_count;
			CheckCudaErrors(cudaMemcpy(column_output->data, &result, dtype_size, cudaMemcpyHostToDevice));
		}
		else if (Ral::Traits::is_dtype_float64(dtype)) {
			double result = (double) avg_sum / (double) avg_count;
			CheckCudaErrors(cudaMemcpy(column_output->data, &result, dtype_size, cudaMemcpyHostToDevice));
		}
		else if (Ral::Traits::is_dtype_integer(dtype)) {
			if (Ral::Traits::is_dtype_signed(dtype)) {
				int64_t result = (int64_t) avg_sum / (int64_t) avg_count;
				CheckCudaErrors(cudaMemcpy(column_output->data, &result, dtype_size, cudaMemcpyHostToDevice));
			}
			//TODO felipe percy noboa see upgrade to uints
			//            else if (Ral::Traits::is_dtype_unsigned(dtype)) {
				//                uint64_t result = (uint64_t) avg_sum / (uint64_t) avg_count;
				//                CheckCudaErrors(cudaMemcpy(column_output->data, &result, dtype_size, cudaMemcpyHostToDevice));
			//            }
		}
		else {
			error = GDF_UNSUPPORTED_DTYPE;
		}
	}
	return error;
}

project_plan_params parse_project_plan(blazing_frame& input, std::string query_part) {

	gdf_error err = GDF_SUCCESS;

	// std::cout<<"starting process_project"<<std::endl;

	size_t size = input.get_column(0).size();


	// LogicalProject(x=[$0], y=[$1], z=[$2], e=[$3], join_x=[$4], y0=[$5], EXPR$6=[+($0, $5)])
	const std::string combined_expression = query_part.substr(
			query_part.find("(") + 1,
			(query_part.rfind(")") - query_part.find("(")) - 1
	);

	std::vector<std::string> expressions = get_expressions_from_expression_list(combined_expression);

	//now we have a vector
	//x=[$0
	std::vector<bool> input_used_in_output(input.get_width(),false);

	std::vector<gdf_column_cpp> columns(expressions.size());
	std::vector<std::string> names(expressions.size());


	std::vector<column_index_type> final_output_positions;
	std::vector<gdf_column *> output_columns;
	std::vector<gdf_column *> input_columns;

	//TODO: some of this code could be used to extract columns
	//that will be projected to make the csv and parquet readers
	//be able to ignore columns that are not
	gdf_dtype max_temp_type = GDF_invalid;
	std::vector<gdf_dtype> output_type_expressions(expressions.size()); //contains output types for columns that are expressions, if they are not expressions we skip over it

	size_t num_expressions_out = 0;
	std::vector<bool> input_used_in_expression(input.get_size_columns(),false);

	for(int i = 0; i < expressions.size(); i++){ //last not an expression
		std::string expression = expressions[i].substr(
				expressions[i].find("=[") + 2 ,
				(expressions[i].size() - expressions[i].find("=[")) - 3
		);

		std::string name = expressions[i].substr(
				0, expressions[i].find("=[")
		);

		if(contains_evaluation(expression)){
			gdf_error err = get_output_type_expression(&input, &output_type_expressions[i], &max_temp_type, expression);
			if(err != GDF_SUCCESS){
				//lets do something some day!!!
			}
			//todo put this into its own function
			std::string clean_expression = clean_calcite_expression(expression);
			int position = clean_expression.size();
			while(position > 0){
				std::string token = get_last_token(clean_expression,&position);

				if(!is_operator_token(token) && !is_literal(token)){
					size_t index = get_index(token);
					input_used_in_expression[index] = true;
				}
			}
			num_expressions_out++;
		}
	}

	//create allocations for output on seperate thread

	std::vector<column_index_type> new_column_indices(input_used_in_expression.size());
	size_t input_columns_used = 0;
	for(int i = 0; i < input_used_in_expression.size(); i++){
		if(input_used_in_expression[i]){
			new_column_indices[i] = input_columns_used;
			input_columns.push_back( input.get_column(i).get_gdf_column());
			input_columns_used++;

		}else{
			new_column_indices[i] = -1; //won't be uesd anyway
		}
	}

	//TODO: this shit is all super hacky in here we should clean it up
	column_index_type temp_cur_out = 0;
	for(int i = 0; i < expressions.size(); i++){ //last not an expression
		std::string expression = expressions[i].substr(
				expressions[i].find("=[") + 2 ,
				(expressions[i].size() - expressions[i].find("=[")) - 3
		);

		std::string name = expressions[i].substr(
				0, expressions[i].find("=[")
		);

		if(contains_evaluation(expression)){
			final_output_positions.push_back(input_columns_used + temp_cur_out);
			temp_cur_out++;
		}
	}

	std::vector<column_index_type>  left_inputs;
	std::vector<column_index_type>  right_inputs;
	std::vector<column_index_type>  outputs;

	std::vector<gdf_binary_operator>  operators;
	std::vector<gdf_unary_operator>  unary_operators;


	std::vector<gdf_scalar>  left_scalars;
	std::vector<gdf_scalar>  right_scalars;
	size_t cur_expression_out = 0;
	for(int i = 0; i < expressions.size(); i++){ //last not an expression
		std::string expression = expressions[i].substr(
				expressions[i].find("=[") + 2 ,
				(expressions[i].size() - expressions[i].find("=[")) - 3
		);

		std::string name = expressions[i].substr(
				0, expressions[i].find("=[")
		);

		if(contains_evaluation(expression)){
			//assumes worst possible case allocation for output
			//TODO: find a way to know what our output size will be
			gdf_column_cpp output;
			output.create_gdf_column(output_type_expressions[i],size,nullptr,get_width_dtype(output_type_expressions[i]), name);

			output_columns.push_back(output.get_gdf_column());

			gdf_error err = add_expression_to_plan(	input,
					expression,
					cur_expression_out,
					num_expressions_out,
					input_columns_used,
					left_inputs,
					right_inputs,
					outputs,
					operators,
					unary_operators,
					left_scalars,
					right_scalars,
					new_column_indices);
			cur_expression_out++;
			columns[i] = output;


		}else{
			int index = get_index(expression);

			//if the column came over via ipc or was already used
			//we dont want to modify in place


			//			if(input_used_in_output[index] || input.get_column(index).is_ipc()){
			//becuase we already used this we can't just 0 copy it
			//we have to make a copy of it here

			gdf_column_cpp output = input.get_column(index);
			output.delete_set_name(name);
//			std::memcpy(output.get_gdf_column()->col_name, name.c_str(),name.size());
			input_used_in_output[index] = true;
			columns[i] = output;
			//			}else{
			//				input_used_in_output[index] = true;
			//				input.get_column(index).set_name(name);

			//				columns[i] = input.get_column(index);
			//			}
		}
	}

	//free_gdf_column(&temp);
	return project_plan_params {
		num_expressions_out,
		output_columns,
		input_columns,
		left_inputs,
		right_inputs,
		outputs,
		final_output_positions,
		operators,
		unary_operators,
		left_scalars,
		right_scalars,
		new_column_indices,
		columns,
		err
	};
}

gdf_error execute_project_plan(blazing_frame & input, std::string query_part){
	
	gdf_error err = GDF_SUCCESS;

	project_plan_params params = parse_project_plan(input, query_part);
	
	//perform operations
	if(params.num_expressions_out > 0){
		err = perform_operation( params.output_columns,
			params.input_columns,
			params.left_inputs,
			params.right_inputs,
			params.outputs,
			params.final_output_positions,
			params.operators,
			params.unary_operators,
			params.left_scalars,
			params.right_scalars,
			params.new_column_indices);

	}

	input.clear();
	input.add_table(params.columns);

	for(size_t i = 0; i < input.get_width(); i++)
	{
		input.get_column(i).update_null_count();
	}
	
	return err;	
}

gdf_error process_project(blazing_frame & input, std::string query_part){

	// std::cout<<"starting process_project"<<std::endl;

	size_t size = input.get_column(0).size();


	// LogicalProject(x=[$0], y=[$1], z=[$2], e=[$3], join_x=[$4], y0=[$5], EXPR$6=[+($0, $5)])
	const std::string combined_expression = query_part.substr(
			query_part.find("(") + 1,
			(query_part.rfind(")") - query_part.find("(")) - 1
	);

	std::vector<std::string> expressions = get_expressions_from_expression_list(combined_expression);

	//now we have a vector
	//x=[$0
	std::vector<bool> input_used_in_output(size,false);

	std::vector<gdf_column_cpp> columns(expressions.size());
	std::vector<std::string> names(expressions.size());


	gdf_dtype max_temp_type = GDF_invalid;
	std::vector<gdf_dtype> output_type_expressions(expressions.size()); //contains output types for columns that are expressions, if they are not expressions we skip over it

	for(int i = 0; i < expressions.size(); i++){ //last not an expression
		std::string expression = expressions[i].substr(
				expressions[i].find("=[") + 2 ,
				(expressions[i].size() - expressions[i].find("=[")) - 3
		);

		std::string name = expressions[i].substr(
				0, expressions[i].find("=[")
		);

		if(contains_evaluation(expression)){
			gdf_error err = get_output_type_expression(&input, &output_type_expressions[i], &max_temp_type, expression);
			if(err != GDF_SUCCESS){
				//lets do something some day!!!
			}
		}
	}

	
	for(int i = 0; i < expressions.size(); i++){ //last not an expression
		std::string expression = expressions[i].substr(
				expressions[i].find("=[") + 2 ,
				(expressions[i].size() - expressions[i].find("=[")) - 3
		);

		std::string name = expressions[i].substr(
				0, expressions[i].find("=[")
		);

		if(contains_evaluation(expression)){
			//assumes worst possible case allocation for output
			//TODO: find a way to know what our output size will be
			gdf_column_cpp output;
			output.create_gdf_column(output_type_expressions[i],size,nullptr,get_width_dtype(output_type_expressions[i]), name);

			gdf_error err = evaluate_expression(
					input,
					expression,
					output);

			columns[i] = output;

			if(err != GDF_SUCCESS){
				//TODO: clean up everything here so we dont run out of memory
				return err;
			}
		}else{
			int index = get_index(expression);

			//if the column came over via ipc or was already used
			//we dont want to modify in place


			//			if(input_used_in_output[index] || input.get_column(index).is_ipc()){
			//becuase we already used this we can't just 0 copy it
			//we have to make a copy of it here

			gdf_column_cpp output = input.get_column(index).clone(name);

			std::memcpy(output.get_gdf_column()->col_name, name.c_str(),name.size());
			input_used_in_output[index] = true;
			columns[i] = output;
			//			}else{
			//				input_used_in_output[index] = true;
			//				input.get_column(index).set_name(name);

			//				columns[i] = input.get_column(index);
			//			}
		}
	}


	input.clear();
	input.add_table(columns);

	return GDF_SUCCESS;
}

std::string get_named_expression(std::string query_part, std::string expression_name){
	if(query_part.find(expression_name + "=[") == query_part.npos){
		return ""; //expression not found
	}
	int start_position =( query_part.find(expression_name + "=["))+ 2 + expression_name.length();
	int end_position = (query_part.find("]",start_position));
	return query_part.substr(start_position,end_position - start_position);
}


blazing_frame process_join(blazing_frame input, std::string query_part){
	static CodeTimer timer;
	timer.reset();

	size_t size = 0; //libgdf will be handling the outputs for these

	gdf_column_cpp left_indices, right_indices;
	//right now it outputs int32
	//TODO de donde saco el nombre de la columna aqui???
	left_indices.create_gdf_column(GDF_INT32,size,nullptr,sizeof(int), "");
	right_indices.create_gdf_column(GDF_INT32,size,nullptr,sizeof(int), "");

	std::string condition = get_condition_expression(query_part);
	std::string join_type = get_named_expression(query_part,"joinType");


	gdf_error err = evaluate_join(
			condition,
			join_type,
			input,
			left_indices.get_gdf_column(),
			right_indices.get_gdf_column()
	);
	Library::Logging::Logger().logInfo("-> Join sub block 1 took " + std::to_string(timer.getDuration()) + " ms");
	// std::cout<<"Indices are starting!"<<std::endl;
	// print_gdf_column(left_indices.get_gdf_column());
	// print_gdf_column(right_indices.get_gdf_column());
	// std::cout<<"Indices are done!"<<std::endl;



	if(err != GDF_SUCCESS){
		//TODO: clean up everything here so we dont run out of memory
		//return err;
	}
	//the options get interesting here. So if the join nis smaller than the input
	// you could write the output in place, saving time for allocations then shrink later on
	// the simplest solution is to reallocate space and free up the old after copying it over

	timer.reset();
	//a data frame should have two "tables"or groups of columns at this point
	std::vector<gdf_column_cpp> new_columns(input.get_size_columns());
	size_t first_table_end_index = input.get_size_column();
	int column_width;
	for(int column_index = 0; column_index < input.get_size_columns(); column_index++){
		gdf_column_cpp output;

		get_column_byte_width(input.get_column(column_index).get_gdf_column(), &column_width);

		//TODO de donde saco el nombre de la columna aqui???
		output.create_gdf_column(input.get_column(column_index).dtype(),left_indices.size(),nullptr,column_width, input.get_column(column_index).name());

		if(column_index < first_table_end_index)
		{
			//materialize with left_indices
			err = materialize_column(input.get_column(column_index).get_gdf_column(),output.get_gdf_column(),left_indices.get_gdf_column());
			// std::cout<<"left table output"<<std::endl;
			// print_gdf_column(output.get_gdf_column());
		}else{
			//materialize with right indices
			err = materialize_column(input.get_column(column_index).get_gdf_column(),output.get_gdf_column(),right_indices.get_gdf_column());
			// std::cout<<"right table output"<<std::endl;
			// print_gdf_column(output.get_gdf_column());
		}
		if(err != GDF_SUCCESS){
			//TODO: clean up all the resources
			//return err;
		}
		//free_gdf_column(input.get_column(column_index));
		output.update_null_count();

		new_columns[column_index] = output;
	}
	input.clear();
	input.add_table(new_columns);
	Library::Logging::Logger().logInfo("-> Join sub block 2 took " + std::to_string(timer.getDuration()) + " ms");
	return input;
}

blazing_frame process_union(blazing_frame& left, blazing_frame& right, std::string query_part){
	bool isUnionAll = (get_named_expression(query_part, "all") == "true");
	if (!isUnionAll) {
		// throw std::domain_error("UNION is not supported, use UNION ALL");
		return blazing_frame{};
	}	

	// Check same number of columns
	if (left.get_size_column(0) != right.get_size_column(0)) {
		return blazing_frame{};
	}

	// Check columns have the same data type
	size_t ncols = left.get_size_column(0);
	for(size_t i = 0; i < ncols; i++)
	{
		if (left.get_column(i).get_gdf_column()->dtype != right.get_column(i).get_gdf_column()->dtype) {
			return blazing_frame{};
		}
	}

	std::vector<gdf_column_cpp> new_table;
	for(size_t i = 0; i < ncols; i++)
	{
		auto gdf_col_left = left.get_column(i).get_gdf_column();
		auto gdf_col_right = right.get_column(i).get_gdf_column();

		std::vector<gdf_column*> columns;
		columns.push_back(gdf_col_left);
		columns.push_back(gdf_col_right);

		size_t col_total_size = gdf_col_left->size + gdf_col_right->size;
		gdf_column_cpp output_col;
		output_col.create_gdf_column(gdf_col_left->dtype, col_total_size, nullptr, get_width_dtype(gdf_col_left->dtype), left.get_column(i).name());

		gdf_error err = gdf_column_concat(output_col.get_gdf_column(),
				columns.data(),
				columns.size());
		if (err != GDF_SUCCESS)
			return blazing_frame{};

		new_table.push_back(output_col);
	}

	blazing_frame result_frame;
	result_frame.add_table(new_table);

	return result_frame;
}

std::vector<int> get_group_columns(std::string query_part){

	std::string temp_column_string = get_named_expression(query_part,"group");
	if(temp_column_string.size() <= 2){
		return std::vector<int>();
	}
	//now you have somethig like {0, 1}
	temp_column_string = temp_column_string.substr(1,temp_column_string.length() - 2);
	std::vector<std::string> column_numbers_string = StringUtil::split(temp_column_string,",");
	std::vector<int> group_columns(column_numbers_string.size());
	for(int i = 0; i < column_numbers_string.size();i++){
		group_columns[i] = std::stoull (column_numbers_string[i],0);
	}
	return group_columns;
}



gdf_error process_aggregate(blazing_frame & input, std::string query_part){
	/*
	 * 			String sql = "select sum(e), sum(z), x, y from hr.emps group by x , y";
	 * 			generates the following calcite relational algebra
	 * 			LogicalProject(EXPR$0=[$2], EXPR$1=[$3], x=[$0], y=[$1])
	 * 	  	  		LogicalAggregate(group=[{0, 1}], EXPR$0=[SUM($2)], EXPR$1=[SUM($3)])
	 *   				LogicalProject(x=[$0], y=[$1], e=[$3], z=[$2])
	 *     					EnumerableTableScan(table=[[hr, emps]])
	 *
	 * 			As you can see the project following aggregate expects the columns to be grouped by to appear BEFORE the expressions
	 */
	gdf_error err{GDF_SUCCESS};
	{
		auto pos = query_part.find("(") + 1;
		if (pos == std::string::npos) {
			throw std::runtime_error{"process_aggregate, parse error, " + query_part};
		}
		auto count = query_part.length() - pos - 1;
		if (count == 0) {
			throw std::runtime_error{"process_aggregate, parse error, " + query_part};
		}
		query_part = query_part.substr(pos, count);
	}

	//get groups
	std::vector<int> group_columns = get_group_columns(query_part);

	//get aggregations
	std::vector<gdf_agg_op> aggregation_types;
	std::vector<std::string>  aggregation_input_expressions;
	std::vector<std::string>  aggregation_column_assigned_aliases;

	bool expressionFound = true;

	std::vector<std::string> expressions = get_expressions_from_expression_list(query_part);

	for(std::string expr : expressions)
	{
		//std::cout << expr << '\n';
		std::string group_str("group");
		std::string expression = std::regex_replace(expr, std::regex("^ +| +$|( ) +"), "$1");
		if (expression.find("group=") == std::string::npos)
		{
			gdf_agg_op operation;
			err = get_aggregation_operation(expression,&operation);
			aggregation_types.push_back(operation);
			aggregation_input_expressions.push_back(get_string_between_outer_parentheses(expression));

			// if the aggregation has an alias, lets capture it here, otherwise we'll figure out what to call the aggregation based on its input
			if (expression.find("EXPR$") == 0)
				aggregation_column_assigned_aliases.push_back("");
			else
				aggregation_column_assigned_aliases.push_back(expression.substr(0, expression.find("=[")));
		}
	}

	// Group by without aggregation 
	if (aggregation_types.size() == 0) {
		size_t num_group_columns = group_columns.size();
		std::vector<gdf_column*> cols(num_group_columns);
		for(int i = 0; i < num_group_columns; i++){
			cols[i] = input.get_column(i).get_gdf_column();
		}

		size_t nrows = input.get_column(0).size();
		std::vector<gdf_column_cpp> output_columns_group(num_group_columns);
		std::vector<gdf_column*> group_by_columns_ptr_out(num_group_columns);
		for(int i = 0; i < num_group_columns; i++){
			gdf_column_cpp& input_column = input.get_column(i);

			output_columns_group[i].create_gdf_column(input_column.dtype(),nrows,nullptr,get_width_dtype(input_column.dtype()), input_column.name());

			group_by_columns_ptr_out[i] = output_columns_group[i].get_gdf_column();
		}

		gdf_column_cpp index_col;
		index_col.create_gdf_column(GDF_INT32,nrows,nullptr,get_width_dtype(GDF_INT32), "");

		gdf_context ctxt;
		ctxt.flag_nulls_sort_behavior = 0; //  Nulls are are treated as largest
		ctxt.flag_groupby_include_nulls = 1; // Nulls are treated as values in group by keys where NULL == NULL (SQL style)

		err = gdf_group_by_wo_aggregations(num_group_columns,
				cols.data(),
				num_group_columns,
				group_columns.data(),
				group_by_columns_ptr_out.data(),
				index_col.get_gdf_column(),
				&ctxt);

		if (err != GDF_SUCCESS) {
			return err;
		}

		//find the widest possible column
		int widest_column = 0;
		for(int i = 0; i < input.get_width();i++){
			int cur_width;
			get_column_byte_width(input.get_column(i).get_gdf_column(), &cur_width);
			if(cur_width > widest_column){
				widest_column = cur_width;
			}
		}

		gdf_column_cpp temp_output;
		temp_output.create_gdf_column(input.get_column(0).dtype(),index_col.size(),nullptr,widest_column, "");
		for(int i = 0; i < num_group_columns; i++){
			temp_output.set_dtype(output_columns_group[i].dtype());

			err = materialize_column(group_by_columns_ptr_out[i],
					temp_output.get_gdf_column(),
					index_col.get_gdf_column());
			temp_output.update_null_count();
			input.set_column(i,temp_output.clone(input.get_column(i).name()));
		}

		return err;
	}


	std::vector<gdf_dtype> aggregation_input_types;

	size_t size = input.get_column(0).size();
	size_t aggregation_size = group_columns.size() == 0? 1 : size; //if you have no groups you will output onlu one row

	for(int i = 0; i < aggregation_types.size(); i++){
		if(contains_evaluation(aggregation_input_expressions[i])){
			gdf_dtype max_temp_type;
			gdf_error err = get_output_type_expression(&input, &aggregation_input_types[i], &max_temp_type, aggregation_input_expressions[i]);
		}
	}



	std::vector<gdf_column *> group_by_columns_ptr{group_columns.size()};
	std::vector<gdf_column *> group_by_columns_ptr_out{group_columns.size()};
	std::vector<gdf_column_cpp> output_columns_group;
	std::vector<gdf_column_cpp> output_columns_aggregations;

	//TODO: fix this input_column goes out of scope before its used
	//create output here and pass in its pointers to this
	for(int group_columns_index = 0; group_columns_index < group_columns.size(); group_columns_index++){
		gdf_column_cpp input_column = input.get_column(group_columns[group_columns_index]);
		group_by_columns_ptr[group_columns_index] = input_column.get_gdf_column();
		gdf_column_cpp output_group;

		//TODO de donde saco el nombre de la columna aqui???
		output_group.create_gdf_column(input_column.dtype(),size,nullptr,get_width_dtype(input_column.dtype()), input_column.name());
		output_columns_group.push_back(output_group);
		//TODO: we have to do this because the gdf_column is not the same as it gets moved
		//aroudn but the pointers are so you cant use the one that you created you have to use int
		group_by_columns_ptr_out[group_columns_index] = output_columns_group[group_columns_index].get_gdf_column();
	}


	for(int i = 0; i < aggregation_types.size(); i++){
		std::string expression = aggregation_input_expressions[i];
		gdf_column_cpp aggregation_input;
		if(contains_evaluation(expression)){

			//we dont knwo what the size of this input will be so allcoate max size
			//TODO de donde saco el nombre de la columna aqui???
			aggregation_input.create_gdf_column(aggregation_input_types[i],size,nullptr,get_width_dtype(aggregation_input_types[i]),"");

			gdf_error err = evaluate_expression(
					input,
					expression,
					aggregation_input);

			if(err != GDF_SUCCESS){
				//TODO: clean up everything here so we dont run out of memory
				return err;
			}
		}else{
			aggregation_input = input.get_column(get_index(expression));
		}


		gdf_error err;
		gdf_dtype output_type = get_aggregation_output_type(aggregation_input.dtype(),aggregation_types[i], group_columns.size());

		/*
        // The 'gdf_sum' libgdf function requires that all input operands have the same dtype.
        if ((group_columns.size() == 0) && (aggregation_types[i] == GDF_SUM)) {
            output_type = aggregation_input.dtype();
        }
		 */

		gdf_column_cpp output_column;
		// if the aggregation was given an alias lets use it, otherwise we'll name it based on the aggregation and input
		if (aggregation_column_assigned_aliases[i] == "")
			output_column.create_gdf_column(output_type,aggregation_size,nullptr,get_width_dtype(output_type), aggregator_to_string(aggregation_types[i]) + "(" + aggregation_input.name() + ")" );
		else
			output_column.create_gdf_column(output_type,aggregation_size,nullptr,get_width_dtype(output_type), aggregation_column_assigned_aliases[i]);

		output_columns_aggregations.push_back(output_column);

		gdf_context ctxt;
		ctxt.flag_distinct = aggregation_types[i] == GDF_COUNT_DISTINCT ? true : false;
		ctxt.flag_method = GDF_HASH;
		ctxt.flag_sort_result = 1;


		switch(aggregation_types[i]){
		case GDF_SUM:
			if (group_columns.size() == 0) {
				if (aggregation_input.get_gdf_column()->size != 0) {
					err = gdf_sum(aggregation_input.get_gdf_column(), output_column.get_gdf_column()->data, get_width_dtype(output_type));
				}
				else {
					err = create_null_value_gdf_column(0,
							output_type,
							aggregation_size,
							aggregator_to_string(aggregation_types[i]),
							output_column,
							output_columns_aggregations);
				}
			}else{
				//				std::cout<<"before"<<std::endl;
				//				print_gdf_column(output_columns_group[0].get_gdf_column());
				err = gdf_group_by_sum(group_columns.size(),group_by_columns_ptr.data(),aggregation_input.get_gdf_column(),
						nullptr,group_by_columns_ptr_out.data(),output_column.get_gdf_column(),&ctxt);
				//				std::cout<<"after"<<std::endl;
				//				print_gdf_column(output_columns_group[0].get_gdf_column());
				//				std::cout<<"direct "<<(group_by_columns_ptr_out[0] == nullptr)<<std::endl;
				//								print_gdf_column(group_by_columns_ptr_out[0]);
				//								std::cout<<"direct done"<<std::endl;
				//
				//								std::cout<<"output column"<<std::endl;
				//								print_gdf_column(output_column.get_gdf_column());
			}

			if(err == GDF_SUCCESS){
				aggregation_size = output_column.size();
				//so that subsequent iterations won't be too large
			}else{
				//be just as responsible as all the other times we didn't do anything when we got an error bback!
			}
			break;
		case GDF_MIN:
			if(group_columns.size() == 0){
				if (aggregation_input.get_gdf_column()->size != 0) {
					err = gdf_min(aggregation_input.get_gdf_column(), output_column.get_gdf_column()->data, get_width_dtype(output_type));
				}
				else {
					err = create_null_value_gdf_column(0,
							output_type,
							aggregation_size,
							aggregator_to_string(aggregation_types[i]),
							output_column,
							output_columns_aggregations);
				}
			}else{
				err = gdf_group_by_min(group_columns.size(),group_by_columns_ptr.data(),aggregation_input.get_gdf_column(),
						nullptr,group_by_columns_ptr_out.data(),output_column.get_gdf_column(),&ctxt);
			}
			if(err == GDF_SUCCESS){
				aggregation_size = output_column.size();
				//so that subsequent iterations won't be too large
			}else{
				//be just as responsible as all the other times we didn't do anything when we got an error bback!
			}
			break;
		case GDF_MAX:
			if(group_columns.size() == 0){
				if (aggregation_input.get_gdf_column()->size != 0) {
					err = gdf_max(aggregation_input.get_gdf_column(), output_column.get_gdf_column()->data, get_width_dtype(output_type));
				}
				else {
					err = create_null_value_gdf_column(0,
							output_type,
							aggregation_size,
							aggregator_to_string(aggregation_types[i]),
							output_column,
							output_columns_aggregations);
				}
			}else{
				err = gdf_group_by_max(group_columns.size(),group_by_columns_ptr.data(),aggregation_input.get_gdf_column(),
						nullptr,group_by_columns_ptr_out.data(),output_column.get_gdf_column(),&ctxt);
			}
			if(err == GDF_SUCCESS){
				aggregation_size = output_column.size();
				//so that subsequent iterations won't be too large
			}else{
				//be just as responsible as all the other times we didn't do anything when we got an error bback!
			}
			break;
		case GDF_AVG:
			if(group_columns.size() == 0){
				if (aggregation_input.get_gdf_column()->size != 0) {
					err = perform_avg(output_column.get_gdf_column(), aggregation_input.get_gdf_column());
				}
				else {
					err = create_null_value_gdf_column(0,
							output_type,
							aggregation_size,
							aggregator_to_string(aggregation_types[i]),
							output_column,
							output_columns_aggregations);
				}
			}
			else{
				err = gdf_group_by_avg(group_columns.size(),group_by_columns_ptr.data(),aggregation_input.get_gdf_column(),
						nullptr,group_by_columns_ptr_out.data(),output_column.get_gdf_column(),&ctxt);
			}
			if(err == GDF_SUCCESS){
				aggregation_size = output_column.size();
				//so that subsequent iterations won't be too large
			}else{
				//be just as responsible as all the other times we didn't do anything when we got an error bback!
			}
			break;
		case GDF_COUNT:
		case GDF_COUNT_DISTINCT:
			if(group_columns.size() == 0){

                // output dtype is GDF_UINT64
                // defined in 'get_aggregation_output_type' function.
                uint64_t result = aggregation_input.get_gdf_column()->size - aggregation_input.get_gdf_column()->null_count;                
				CheckCudaErrors(cudaMemcpy(output_column.get_gdf_column()->data, &result, sizeof(uint64_t), cudaMemcpyHostToDevice));			

                err = GDF_SUCCESS;

			}else{
				err = gdf_group_by_count(group_columns.size(),group_by_columns_ptr.data(),aggregation_input.get_gdf_column(),
						nullptr,group_by_columns_ptr_out.data(),output_column.get_gdf_column(),&ctxt);
			}
			if(err == GDF_SUCCESS){
				aggregation_size = output_column.size();
				//so that subsequent iterations won't be too large
			}else{
				//be just as responsible as all the other times we didn't do anything when we got an error bback!
			}
			break;

		}

		/*
		 * GDF_SUM = 0,
  GDF_MIN,
  GDF_MAX,
  GDF_AVG,
  GDF_COUNT,
  GDF_COUNT_DISTINCT,
  N_GDF_AGG_OPS
		 */
		//perform aggregation now

		//catpure asize for next iterationo

	}

	//TODO: this is pretty crappy because its recalcluating the groups each time, this is becuase the libgdf api can
	//only process one aggregate at a time while it calculates the group,
	//these steps would have to be divided up in order to really work

	//TODO: consider compacting columns here before moving on
	for(int i = 0; i < output_columns_aggregations.size(); i++){

		output_columns_aggregations[i].resize(aggregation_size);
		output_columns_aggregations[i].compact();
		output_columns_aggregations[i].update_null_count();
	}

	for(int i = 0; i < output_columns_group.size(); i++){
		// print_gdf_column(output_columns_group[i].get_gdf_column());
		output_columns_group[i].resize(aggregation_size);
		output_columns_group[i].compact();
		output_columns_group[i].update_null_count();
	}

	input.clear();

	input.add_table(output_columns_group);
	input.add_table(output_columns_aggregations);
	input.consolidate_tables();

}



gdf_error process_sort(blazing_frame & input, std::string query_part){
	static CodeTimer timer;
	timer.reset();
	std::cout<<"about to process sort"<<std::endl;

	auto rangeStart = query_part.find("(");
	auto rangeEnd = query_part.rfind(")") - rangeStart - 1;
	std::string combined_expression = query_part.substr(rangeStart + 1, rangeEnd - 1);

	//LogicalSort(sort0=[$4], sort1=[$7], dir0=[ASC], dir1=[ASC])
	size_t num_sort_columns = count_string_occurrence(combined_expression,"sort");

	std::vector<int8_t> sort_order_types(num_sort_columns);
	std::vector<gdf_column*> cols(num_sort_columns);
	for(int i = 0; i < num_sort_columns; i++){
		int sort_column_index = get_index(get_named_expression(combined_expression, "sort" + std::to_string(i)));
		cols[i] = input.get_column(sort_column_index).get_gdf_column();

		sort_order_types[i] = (get_named_expression(combined_expression, "dir" + std::to_string(i)) == DESCENDING_ORDER_SORT_TEXT);
	}

	Library::Logging::Logger().logInfo("-> Sort sub block 1 took " + std::to_string(timer.getDuration()) + " ms");
	timer.reset();

	gdf_column_cpp asc_desc_col;
	asc_desc_col.create_gdf_column(GDF_INT8,num_sort_columns,nullptr,1, "");
	CheckCudaErrors(cudaMemcpy(asc_desc_col.get_gdf_column()->data, sort_order_types.data(), sort_order_types.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

	gdf_column_cpp index_col;
	index_col.create_gdf_column(GDF_INT32,input.get_column(0).size(),nullptr,get_width_dtype(GDF_INT32), "");

	gdf_context context;
	context.flag_nulls_sort_behavior = 0; // Nulls are are treated as largest

	gdf_error err = gdf_order_by(cols.data(),
			(int8_t*)(asc_desc_col.get_gdf_column()->data),
			num_sort_columns,
			index_col.get_gdf_column(),
			&context);

	Library::Logging::Logger().logInfo("-> Sort sub block 2 took " + std::to_string(timer.getDuration()) + " ms");

	if (err != GDF_SUCCESS)
		return err;

	timer.reset();
	int widest_column = 0;
	for(int i = 0; i < input.get_width();i++){
		int cur_width;
		get_column_byte_width(input.get_column(i).get_gdf_column(), &cur_width);
		if(cur_width > widest_column){
			widest_column = cur_width;
		}

	}
	//find the widest possible column

	gdf_column_cpp temp_output;
	//TODO de donde saco el nombre de la columna aqui???
	temp_output.create_gdf_column(input.get_column(0).dtype(),input.get_column(0).size(),nullptr,widest_column, "");
	//now we need to materialize
	//i dont think we can do that in place since we are writing and reading out of order
	for(int i = 0; i < input.get_width();i++){
		temp_output.set_dtype(input.get_column(i).dtype());

		gdf_error err = materialize_column(
				input.get_column(i).get_gdf_column(),
				temp_output.get_gdf_column(),
				index_col.get_gdf_column()
		);

		temp_output.update_null_count();
		input.set_column(i,temp_output.clone(input.get_column(i).name()));

		/*gdf_column_cpp empty;

		int width;
		get_column_byte_width(input.get_column(i).get_gdf_column(), &width);

		//TODO de donde saco el nombre de la columna aqui???
		empty.create_gdf_column(input.get_column(i).dtype(),0,nullptr,width, "");

		//copy output back to dat aframe

		gdf_column_cpp new_output;
		if(input.get_column(i).is_ipc()){
			//TODO de donde saco el nombre de la columna aqui???
			new_output.create_gdf_column(input.get_column(i).dtype(), input.get_column(i).size(),nullptr,get_width_dtype(input.get_column(i).dtype()), "");
			input.set_column(i,new_output);
		}else{
			new_output = input.get_column(i);
		}
		err = gpu_concat(temp_output.get_gdf_column(), empty.get_gdf_column(), new_output.get_gdf_column());

		//free_gdf_column(&empty);*/
	}
	Library::Logging::Logger().logInfo("-> Sort sub block 3 took " + std::to_string(timer.getDuration()) + " ms");
	return GDF_SUCCESS;
}

//TODO: this does not compact the allocations which would be nice if it could
gdf_error process_filter(blazing_frame & input, std::string query_part){

	//assert(input.get_column(0) != nullptr);
	static CodeTimer timer;

	size_t size = input.get_column(0).size();

	timer.reset();

	//TODO de donde saco el nombre de la columna aqui???
	gdf_column_cpp stencil;
	stencil.create_gdf_column(GDF_INT8,input.get_column(0).size(),nullptr,1, "");

	gdf_dtype output_type_junk; //just gets thrown away
	gdf_dtype max_temp_type = GDF_INT8;
	for(int i = 0; i < input.get_width(); i++){
		if(get_width_dtype(input.get_column(i).dtype()) > get_width_dtype(max_temp_type)){
			max_temp_type = input.get_column(i).dtype();
		}
	}

	Library::Logging::Logger().logInfo("-> Filter sub block 1 took " + std::to_string(timer.getDuration()) + " ms");
	timer.reset();
	gdf_dtype output_type; // this is junk since we know the output types here
	gdf_error err = get_output_type_expression(&input, &output_type, &max_temp_type, get_condition_expression(query_part));
	if(err != GDF_SUCCESS){
		//panic then do wonderful things here to fix everything
		//im really liking Andrescus talk on control flow blah blah something i forget his name
	}

	Library::Logging::Logger().logInfo("-> Filter sub block 2 took " + std::to_string(timer.getDuration()) + " ms");

	timer.reset();
	std::string conditional_expression = get_condition_expression(query_part);
	Library::Logging::Logger().logInfo("-> Filter sub block 3 took " + std::to_string(timer.getDuration()) + " ms");
	// timer.reset();
	err = evaluate_expression(
			input,
			conditional_expression,
			stencil);

	// Library::Logging::Logger().logInfo("-> Filter sub block 4 took " + std::to_string(timer.getDuration()) + " ms");

	if(err == GDF_SUCCESS){
		//apply filter to all the columns
		// for(int i = 0; i < input.get_width(); i++){
		// 	temp.create_gdf_column(input.get_column(i).dtype(), input.get_column(i).size(), nullptr, get_width_dtype(input.get_column(i).dtype()));
		// 	//temp.set_dtype(input.get_column(i).dtype());

		// 	//			cudaPointerAttributes attributes;
		// 	//			cudaError_t err2 = cudaPointerGetAttributes ( &attributes, (void *) temp.data );
		// 	//			err2 = cudaPointerGetAttributes ( &attributes, (void *) input.get_column(i)->data );
		// 	//			err2 = cudaPointerGetAttributes ( &attributes, (void *) stencil.data );


		// 	//just for testing
		// 	//			cudaMalloc((void **)&(temp.data),1000);
		// 	//			cudaMalloc((void **)&(temp.valid),1000);

		// 	err = gpu_apply_stencil(
		// 			input.get_column(i).get_gdf_column(),
		// 			stencil.get_gdf_column(),
		// 			temp.get_gdf_column()
		// 	);
		// 	if(err != GDF_SUCCESS){
		// 		return err;
		// 	}


		// 	input.set_column(i,temp.clone());
		// }

		timer.reset();
		gdf_column_cpp index_col;
		index_col.create_gdf_column(GDF_INT32,input.get_column(0).size(),nullptr,get_width_dtype(GDF_INT32), "");
		gdf_sequence(static_cast<int32_t*>(index_col.get_gdf_column()->data), input.get_column(0).size(), 0);
		// std::vector<int32_t> idx(input.get_column(0).size());
		// std::iota(idx.begin(),idx.end(),0);
		// CheckCudaErrors(cudaMemcpy(index_col.get_gdf_column()->data, idx.data(), idx.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
		Library::Logging::Logger().logInfo("-> Filter sub block 5 took " + std::to_string(timer.getDuration()) + " ms");

		gdf_column_cpp temp_idx;
		temp_idx.create_gdf_column(GDF_INT32, input.get_column(0).size(), nullptr, get_width_dtype(GDF_INT32));

		timer.reset();
		err = gpu_apply_stencil(
				index_col.get_gdf_column(),
				stencil.get_gdf_column(),
				temp_idx.get_gdf_column()
		);
		Library::Logging::Logger().logInfo("-> Filter sub block 6 took " + std::to_string(timer.getDuration()) + " ms");
		if(err != GDF_SUCCESS){
			return err;
		}

		timer.reset();
		gdf_column_cpp materialize_temp;
		materialize_temp.create_gdf_column(input.get_column(0).dtype(),temp_idx.size(),nullptr,get_width_dtype(max_temp_type), "");
		for(int i = 0; i < input.get_width();i++){
			materialize_temp.set_dtype(input.get_column(i).dtype());

			gdf_error err = materialize_column(
					input.get_column(i).get_gdf_column(),
					materialize_temp.get_gdf_column(),
					temp_idx.get_gdf_column()
			);

			materialize_temp.update_null_count();
			input.set_column(i,materialize_temp.clone(input.get_column(i).name()));
		}
		Library::Logging::Logger().logInfo("-> Filter sub block 7 took " + std::to_string(timer.getDuration()) + " ms");
	}else{
		//free_gdf_column(&stencil);
		//free_gdf_column(&temp);
		return err;
	}
	//free_gdf_column(&stencil);
	//free_gdf_column(&temp);
	return GDF_SUCCESS;

}

//Returns the index from table if exists
size_t get_table_index(std::vector<std::string> table_names, std::string table_name){
	std::cout << "---> BEGIN: get_table_index\n";
	std::cout << "Table: "<< table_name << "\n";
	std::cout << "Catalog of tables\n";

	for (auto tb : table_names) {
		std::cout << tb << "\n";
	}

	auto it = std::find(table_names.begin(), table_names.end(), table_name);
	if(it != table_names.end()){
		return std::distance(table_names.begin(), it);
	}else{
		throw std::invalid_argument( "table name does not exists" );
	}
}

//TODO: if a table needs to be used more than once you need to include it twice
//i know that kind of sucks, its for the 0 copy stuff, this can easily be remedied
//by changings scan to make copies
blazing_frame evaluate_split_query(
		std::vector<std::vector<gdf_column_cpp> > input_tables,
		std::vector<std::string> table_names,
		std::vector<std::vector<std::string>> column_names,
		std::vector<std::string> query, int call_depth = 0){
	assert(input_tables.size() == table_names.size());

	static CodeTimer blazing_timer;			

	if(query.size() == 1){
		//process yourself and return

		if(is_scan(query[0])){
			blazing_frame scan_frame;
			//EnumerableTableScan(table=[[hr, joiner]])
			scan_frame.add_table(
					input_tables[
					             get_table_index(
					            		 table_names,
					            		 extract_table_name(query[0])
					             )
					             ]
			);
			return scan_frame;
		}else{
			//i dont think there are any other type of end nodes at the moment
		}
	}

	if(is_double_input(query[0])){

		int other_depth_one_start = 2;
		for(int i = 2; i < query.size(); i++){
			int j = 0;
			while(query[i][j] == ' '){

				j+=2;
			}
			int depth = (j / 2) - call_depth;
			if(depth == 1){
				other_depth_one_start = i;
			}
		}
		//these shoudl be split up and run on different threads
		blazing_frame left_frame;
		left_frame = evaluate_split_query(
				input_tables,
				table_names,
				column_names,
				std::vector<std::string>(
						query.begin() + 1,
						query.begin() + other_depth_one_start),
						call_depth + 1
		);

		blazing_frame right_frame;
		right_frame = evaluate_split_query(
				input_tables,
				table_names,
				column_names,
				std::vector<std::string>(
						query.begin() + other_depth_one_start,
						query.end()),
						call_depth + 1
		);

		blazing_frame result_frame;
		if(is_join(query[0])){
			//we know that left and right are dataframes we want to join together
			left_frame.add_table(right_frame.get_columns()[0]);
			///left_frame.consolidate_tables();
			blazing_timer.reset();
			result_frame = process_join(left_frame,query[0]);
			Library::Logging::Logger().logInfo("process_join took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(left_frame.get_column(0).size()) + " rows with an output of " + std::to_string(result_frame.get_column(0).size()));
			return result_frame;
		}else if(is_union(query[0])){
			//TODO: append the frames to each other
			//return right_frame;//!!
			blazing_timer.reset();
			result_frame = process_union(left_frame,right_frame,query[0]);
			Library::Logging::Logger().logInfo("process_union took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(left_frame.get_column(0).size()) + " rows with an output of " + std::to_string(result_frame.get_column(0).size()));
			return result_frame;
		}else{
			//probably an error here
		}

	}else{
		//process child
		blazing_frame child_frame = evaluate_split_query(
				input_tables,
				table_names,
				column_names,
				std::vector<std::string>(
						query.begin() + 1,
						query.end()),
						call_depth + 1
		);
		//process self
		if(is_project(query[0])){
			blazing_timer.reset();
			gdf_error err = execute_project_plan(child_frame,query[0]);
			Library::Logging::Logger().logInfo("process_project took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_column(0).size()) + " rows");
			return child_frame;
		}else if(is_aggregate(query[0])){
			blazing_timer.reset();
			gdf_error err = process_aggregate(child_frame,query[0]);
			Library::Logging::Logger().logInfo("process_aggregate took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_column(0).size()) + " rows");
			return child_frame;
		}else if(is_sort(query[0])){
			blazing_timer.reset();
			gdf_error err = process_sort(child_frame,query[0]);
			Library::Logging::Logger().logInfo("process_sort took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_column(0).size()) + " rows");
			return child_frame;
		}else if(is_filter(query[0])){
			blazing_timer.reset();
			gdf_error err = process_filter(child_frame,query[0]);
			Library::Logging::Logger().logInfo("process_filter took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_column(0).size()) + " rows");
			if(err != GDF_SUCCESS){
				std::cout<<"Error in filter: "<<err<<std::endl;
			}

			return child_frame;
		}else{
			//some error
		}
		//return frame
	}
}

query_token_t evaluate_query(
		std::vector<std::vector<gdf_column_cpp> > input_tables,
		std::vector<std::string> table_names,
		std::vector<std::vector<std::string>> column_names,
		std::string logicalPlan,
		connection_id_t connection,
		std::vector<void *> handles){

	std::cout<<"Input\n";
	//	print_column<int8_t>(input_tables[0][0].get_gdf_column());

	query_token_t token = result_set_repository::get_instance().register_query(connection); //register the query so we can receive result requests for it

	std::thread t = std::thread([=]{

		CodeTimer blazing_timer;

		std::vector<std::string> splitted = StringUtil::split(logicalPlan, "\n");
		if (splitted[splitted.size() - 1].length() == 0) {
			splitted.erase(splitted.end() -1);
		}


		blazing_frame output_frame = evaluate_split_query(input_tables, table_names, column_names, splitted);

		//REMOVE any columns that were ipcd to put into the result set
		std::set<gdf_column *> included_columns;
		for(size_t index = 0; index < output_frame.get_size_columns(); index++){
			gdf_column_cpp output_column = output_frame.get_column(index);
			output_frame.set_column(index, output_column.clone(output_column.name()));
		}

		//Todo: put it on a macro for debugging purposes!
		/*std::cout<<"Result\n";
	for (auto outputTable : output_frame.get_columns()) {
		for (auto outputColumn : outputTable) {
			print_gdf_column(outputColumn.get_gdf_column());
		}
	}
	std::cout<<"end:Result\n";*/

		double duration = blazing_timer.getDuration();
		result_set_repository::get_instance().update_token(token, output_frame, duration);

		//@todo: hablar con felipe sobre cudaIpcCloseMemHandle
		for(int i = 0; i < handles.size(); i++){
			cudaIpcCloseMemHandle (handles[i]);
		}
		//			std::cout<<"Result\n";
		//			print_column<int8_t>(output_frame.get_columns()[0][0].get_gdf_column());
	});;

	//@todo: hablar con felipe sobre detach
	t.detach();

	return token;
}

gdf_error evaluate_query(
		std::vector<std::vector<gdf_column_cpp> > input_tables,
		std::vector<std::string> table_names,
		std::vector<std::vector<std::string>> column_names,
		std::string logicalPlan,
		std::vector<gdf_column_cpp> & outputs){

	std::vector<std::string> splitted = StringUtil::split(logicalPlan, "\n");
	if (splitted[splitted.size() - 1].length() == 0) {
		splitted.erase(splitted.end() -1);
	}
	blazing_frame output_frame = evaluate_split_query(input_tables, table_names, column_names, splitted);

	for(size_t i=0;i<output_frame.get_width();i++){

		GDFRefCounter::getInstance()->deregister_column(output_frame.get_column(i).get_gdf_column());
		outputs.push_back(output_frame.get_column(i));

	}

	return GDF_SUCCESS;
}
