#include "CalciteInterpreter.h"

#include <blazingdb/io/Library/Logging/Logger.h>
#include <blazingdb/io/Util/StringUtil.h>

#include <algorithm>
#include <thread>
#include <regex>
#include <string>
#include <set>

#include "config/GPUManager.cuh"
#include "Utils.cuh"
#include "LogicalFilter.h"
#include "ResultSetRepository.h"
#include "JoinProcessor.h"
#include "ColumnManipulation.cuh"
#include "CalciteExpressionParsing.h"
#include "CodeTimer.h"
#include "Traits/RuntimeTraits.h"
#include "Interpreter/interpreter_cpp.h"
#include "operators/OrderBy.h"
#include "operators/GroupBy.h"
#include "operators/JoinOperator.h"
#include "utilities/RalColumn.h"
#include "io/DataLoader.h"
#include "reduction.hpp"
#include "stream_compaction.hpp"
#include "groupby.hpp"
#include <cudf/legacy/table.hpp>
#include "cudf/binaryop.hpp"
#include <rmm/thrust_rmm_allocator.h>

const std::string LOGICAL_JOIN_TEXT = "LogicalJoin";
const std::string LOGICAL_UNION_TEXT = "LogicalUnion";
const std::string LOGICAL_SCAN_TEXT = "LogicalTableScan";
const std::string BINDABLE_SCAN_TEXT = "BindableTableScan";
const std::string LOGICAL_AGGREGATE_TEXT = "LogicalAggregate";
const std::string LOGICAL_PROJECT_TEXT = "LogicalProject";
const std::string LOGICAL_SORT_TEXT = "LogicalSort";
const std::string LOGICAL_FILTER_TEXT = "LogicalFilter";
const std::string ASCENDING_ORDER_SORT_TEXT = "ASC";
const std::string DESCENDING_ORDER_SORT_TEXT = "DESC";

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


bool is_logical_scan(std::string query_part){
	return (query_part.find(LOGICAL_SCAN_TEXT) != std::string::npos);
}

bool is_bindable_scan(std::string query_part){
	return (query_part.find(BINDABLE_SCAN_TEXT) != std::string::npos);
}

bool is_scan(std::string query_part){
	return is_logical_scan(query_part) || is_bindable_scan(query_part);
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
	if(ral::operators::is_join(query_part)){
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

project_plan_params parse_project_plan(blazing_frame& input, std::string query_part) {

	gdf_error err = GDF_SUCCESS;

	// std::cout<<"starting process_project"<<std::endl;

	size_t size = input.get_num_rows_in_table(0);


	// LogicalProject(x=[$0], y=[$1], z=[$2], e=[$3], join_x=[$4], y0=[$5], EXPR$6=[+($0, $5)])
	std::string combined_expression = query_part.substr(
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
			output_type_expressions[i] = get_output_type_expression(&input, &max_temp_type, expression);

			//todo put this into its own function
			std::string clean_expression = clean_calcite_expression(expression);
			
			std::vector<std::string> tokens = get_tokens_in_reverse_order(clean_expression);
			for (std::string token : tokens){
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

			add_expression_to_plan(	input,
					input_columns,
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
			output.set_name(name);
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

void execute_project_plan(blazing_frame & input, std::string query_part){
	project_plan_params params = parse_project_plan(input, query_part);

	//perform operations
	if(params.num_expressions_out > 0){
		perform_operation( params.output_columns,
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
}

void process_project(blazing_frame & input, std::string query_part){

	// // std::cout<<"starting process_project"<<std::endl;

	// size_t size = input.get_num_rows_in_table(0);

	// // LogicalProject(x=[$0], y=[$1], z=[$2], e=[$3], join_x=[$4], y0=[$5], EXPR$6=[+($0, $5)])
	// std::string combined_expression = query_part.substr(
	// 		query_part.find("(") + 1,
	// 		(query_part.rfind(")") - query_part.find("(")) - 1
	// );

	// std::vector<std::string> expressions = get_expressions_from_expression_list(combined_expression);

	// //now we have a vector
	// //x=[$0
	// std::vector<bool> input_used_in_output(size,false);

	// std::vector<gdf_column_cpp> columns(expressions.size());
	// std::vector<std::string> names(expressions.size());

	// gdf_dtype max_temp_type = GDF_invalid;
	// std::vector<gdf_dtype> output_type_expressions(expressions.size()); //contains output types for columns that are expressions, if they are not expressions we skip over it

	// for(int i = 0; i < expressions.size(); i++){ //last not an expression
	// 	std::string expression = expressions[i].substr(
	// 			expressions[i].find("=[") + 2 ,
	// 			(expressions[i].size() - expressions[i].find("=[")) - 3
	// 	);

	// 	std::string name = expressions[i].substr(
	// 			0, expressions[i].find("=[")
	// 	);

	// 	if(contains_evaluation(expression)){
	// 		output_type_expressions[i] = get_output_type_expression(&input, &max_temp_type, expression);
	// 	}
	// }


	// for(int i = 0; i < expressions.size(); i++){ //last not an expression
	// 	std::string expression = expressions[i].substr(
	// 			expressions[i].find("=[") + 2 ,
	// 			(expressions[i].size() - expressions[i].find("=[")) - 3
	// 	);

	// 	std::string name = expressions[i].substr(
	// 			0, expressions[i].find("=[")
	// 	);

	// 	if(contains_evaluation(expression)){
	// 		//assumes worst possible case allocation for output
	// 		//TODO: find a way to know what our output size will be
	// 		gdf_column_cpp output;
	// 		output.create_gdf_column(output_type_expressions[i],size,nullptr,get_width_dtype(output_type_expressions[i]), name);

	// 		evaluate_expression(input, expression,  output);

	// 		columns[i] = output;
	// 	}else{
	// 		int index = get_index(expression);

	// 		//if the column came over via ipc or was already used
	// 		//we dont want to modify in place


	// 		//			if(input_used_in_output[index] || input.get_column(index).is_ipc()){
	// 		//becuase we already used this we can't just 0 copy it
	// 		//we have to make a copy of it here

	// 		gdf_column_cpp output = input.get_column(index).clone(name);
	// 		input_used_in_output[index] = true;
	// 		columns[i] = output;
	// 		//			}else{
	// 		//				input_used_in_output[index] = true;
	// 		//				input.get_column(index).set_name(name);

	// 		//				columns[i] = input.get_column(index);
	// 		//			}
	// 	}
	// }

	// input.clear();
	// input.add_table(columns);

}


std::string get_named_expression(std::string query_part, std::string expression_name){
	if(query_part.find(expression_name + "=[") == query_part.npos){
		return ""; //expression not found
	}
	int start_position =( query_part.find(expression_name + "=[["))+ 3 + expression_name.length();
	if (query_part.find(expression_name + "=[[") == query_part.npos){
		start_position =( query_part.find(expression_name + "=["))+ 2 + expression_name.length();
	}
	int end_position = (query_part.find("]",start_position));
	return query_part.substr(start_position,end_position - start_position);
}

std::string get_condition_expression(std::string query_part){
    return get_named_expression(query_part,"condition");
}



blazing_frame process_union(blazing_frame& left, blazing_frame& right, std::string query_part){
	bool isUnionAll = (get_named_expression(query_part, "all") == "true");
	if (!isUnionAll) {
		throw std::runtime_error{"In process_union function: UNION is not supported, use UNION ALL"};
	}

	// Check same number of columns
	if (left.get_size_column(0) != right.get_size_column(0)) {
		throw std::runtime_error{"In process_union function: left frame and right frame have different number of columns"};
	}

	// Check columns have the same data type
	size_t ncols = left.get_size_column(0);
	for(size_t i = 0; i < ncols; i++)
	{
		if (left.get_column(i).get_gdf_column()->dtype != right.get_column(i).get_gdf_column()->dtype) {
			throw std::runtime_error{"In process_union function: left column and right column have different dtypes"};
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

		CUDF_CALL( gdf_column_concat(output_col.get_gdf_column(),
										  columns.data(),
										  columns.size()) );

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
	std::vector<int> group_columns_indices(column_numbers_string.size());
	for(int i = 0; i < column_numbers_string.size();i++){
		group_columns_indices[i] = std::stoull (column_numbers_string[i],0);
	}
	return group_columns_indices;
}


//TODO: this does not compact the allocations which would be nice if it could
void process_filter(blazing_frame & input, std::string query_part){
	static CodeTimer timer;

	size_t size = input.get_num_rows_in_table(0);

	timer.reset();

	if (size > 0){
		
		//TODO de donde saco el nombre de la columna aqui???
		gdf_column_cpp stencil;
		stencil.create_gdf_column(GDF_INT8,input.get_num_rows_in_table(0),nullptr,1, "");

		Library::Logging::Logger().logInfo("-> Filter sub block 1 took " + std::to_string(timer.getDuration()) + " ms");
		timer.reset();

		std::string conditional_expression = get_condition_expression(query_part);
		evaluate_expression(input, conditional_expression, stencil);

		Library::Logging::Logger().logInfo("-> Filter sub block 3 took " + std::to_string(timer.getDuration()) + " ms");
		timer.reset();
		
		gdf_column_cpp index_col;
		index_col.create_gdf_column(GDF_INT32,input.get_num_rows_in_table(0),nullptr,get_width_dtype(GDF_INT32), "");
		gdf_sequence(static_cast<int32_t*>(index_col.get_gdf_column()->data), input.get_num_rows_in_table(0), 0);
		// std::vector<int32_t> idx(input.get_num_rows_in_table(0));
		// std::iota(idx.begin(),idx.end(),0);
		// CheckCudaErrors(cudaMemcpy(index_col.get_gdf_column()->data, idx.data(), idx.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
		Library::Logging::Logger().logInfo("-> Filter sub block 5 took " + std::to_string(timer.getDuration()) + " ms");


		timer.reset();
		stencil.get_gdf_column()->dtype = GDF_BOOL8; // apply_boolean_mask expects the stencil to be a GDF_BOOL8 which for our purposes the way we are using the GDF_INT8 is the same as GDF_BOOL8

		cudf::table inputToFilter = ral::utilities::create_table(input.get_columns()[0]);
		cudf::table filteredData = cudf::apply_boolean_mask(inputToFilter, *(stencil.get_gdf_column()));

		for(int i = 0; i < input.get_width();i++){
			gdf_column* temp_col_view = filteredData.get_column(i);
			temp_col_view->col_name = nullptr; // lets do this because its not always set properly
			gdf_column_cpp temp;
			temp.create_gdf_column(filteredData.get_column(i));
			temp.set_name(input.get_column(i).name());
			input.set_column(i,temp);
		}

		// gdf_column temp_idx = cudf::apply_boolean_mask( *(index_col.get_gdf_column()), *(stencil.get_gdf_column()));
		// temp_idx.col_name = nullptr;
		// gdf_column * temp_idx_ptr = new gdf_column;
		// *temp_idx_ptr = temp_idx;
		// gdf_column_cpp temp_idx_col;
		// temp_idx_col.create_gdf_column(temp_idx_ptr);

		// Library::Logging::Logger().logInfo("-> Filter sub block 6 took " + std::to_string(timer.getDuration()) + " ms");

		// timer.reset();
		
		// for(int i = 0; i < input.get_width();i++){
		// 	gdf_column_cpp materialize_temp;
		// 	if (input.get_column(i).valid())
		// 		materialize_temp.create_gdf_column(input.get_column(i).dtype(),temp_idx_col.size(),nullptr,get_width_dtype(input.get_column(i).dtype()), input.get_column(i).name());
		// 	else
		// 		materialize_temp.create_gdf_column(input.get_column(i).dtype(),temp_idx_col.size(),nullptr,nullptr,get_width_dtype(input.get_column(i).dtype()), input.get_column(i).name());

		// 	materialize_column(
		// 			input.get_column(i).get_gdf_column(),
		// 			materialize_temp.get_gdf_column(), //output
		// 			temp_idx_col.get_gdf_column() //indexes
		// 	);	
		// 	input.set_column(i,materialize_temp);
		// }
	}
	Library::Logging::Logger().logInfo("-> Filter sub block 7 took " + std::to_string(timer.getDuration()) + " ms");
}

//Returns the index from table if exists
size_t get_table_index(std::vector<std::string> table_names, std::string table_name){
	std::cout << "---> BEGIN: get_table_index\n";
	std::cout << "Table: "<< table_name << "\n";
	std::cout << "Catalog of tables\n";

	for (auto tb : table_names) {
		std::cout << tb << "\n";
	}

	if (StringUtil::beginsWith(table_name, "main.")){
		table_name = table_name.substr(5);
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
		std::vector<std::string> query,
		const Context* queryContext,
		int call_depth = 0){
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
						queryContext,
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
						queryContext,
						call_depth + 1
		);

		blazing_frame result_frame;

		if (ral::operators::is_join(query[0])) {
			//we know that left and right are dataframes we want to join together
			left_frame.add_table(right_frame.get_columns()[0]);
			///left_frame.consolidate_tables();
			blazing_timer.reset();
			result_frame = ral::operators::process_join(queryContext, left_frame, query[0]);
			Library::Logging::Logger().logInfo("process_join took " + std::to_string(blazing_timer.getDuration()) + " ms with an output of " + std::to_string(result_frame.get_num_rows_in_table(0)));
			return result_frame;
		}else if(is_union(query[0])){
			//TODO: append the frames to each other
			//return right_frame;//!!
			blazing_timer.reset();
			result_frame = process_union(left_frame,right_frame,query[0]);
			Library::Logging::Logger().logInfo("process_union took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(left_frame.get_num_rows_in_table(0)) + " rows with an output of " + std::to_string(result_frame.get_num_rows_in_table(0)));
			return result_frame;
		}else{
			throw std::runtime_error{"In evaluate_split_query function: unsupported query operator"};
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
				queryContext,
				call_depth + 1
		);
		//process self
		if(is_project(query[0])){
			blazing_timer.reset();
			execute_project_plan(child_frame,query[0]);
			Library::Logging::Logger().logInfo("process_project took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_num_rows_in_table(0)) + " rows");
			return child_frame;
		}else if(ral::operators::is_aggregate(query[0])){
			blazing_timer.reset();
			ral::operators::process_aggregate(child_frame, query[0], queryContext);
			Library::Logging::Logger().logInfo("process_aggregate took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_num_rows_in_table(0)) + " rows");
			return child_frame;
		}else if(ral::operators::is_sort(query[0])){
			blazing_timer.reset();
			ral::operators::process_sort(child_frame, query[0], queryContext);
			Library::Logging::Logger().logInfo("process_sort took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_num_rows_in_table(0)) + " rows");
			return child_frame;
		}else if(is_filter(query[0])){
			blazing_timer.reset();
			process_filter(child_frame,query[0]);
			Library::Logging::Logger().logInfo("process_filter took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_num_rows_in_table(0)) + " rows");
			return child_frame;
		}else{
			throw std::runtime_error{"In evaluate_split_query function: unsupported query operator"};
		}
		//return frame
	}
}

blazing_frame evaluate_split_query(
		std::vector<ral::io::data_loader > input_loaders,
		std::vector<ral::io::Schema> schemas,
		std::vector<std::string> table_names,
		std::vector<std::string> query, 
		const Context* queryContext, 
		int call_depth = 0){
	assert(input_loaders.size() == table_names.size());

	static CodeTimer blazing_timer;

	if(query.size() == 1){
		//process yourself and return

		if(is_scan(query[0])){
			blazing_frame scan_frame;
			std::vector<gdf_column_cpp>  input_table;

			size_t table_index =  get_table_index(
				            		 table_names,
				            		 extract_table_name(query[0]));
			if(is_bindable_scan(query[0])){
				std::string project_string = get_named_expression(query[0],"projects");
				std::vector<std::string> project_string_split = get_expressions_from_expression_list(project_string, true);
				std::vector<size_t> projections;
				for(int i = 0; i < project_string_split.size(); i++){
					projections.push_back(std::stoull(project_string_split[i]));
				}
				input_loaders[table_index].load_data(input_table,projections,schemas[table_index]);
			}else{
				input_loaders[table_index].load_data(input_table,{},schemas[table_index]);
			}


			//EnumerableTableScan(table=[[hr, joiner]])
			scan_frame.add_table(input_table);
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
				input_loaders,
				schemas,
				table_names,
				std::vector<std::string>(
						query.begin() + 1,
						query.begin() + other_depth_one_start),
						queryContext,
						call_depth + 1
		);

		blazing_frame right_frame;
		right_frame = evaluate_split_query(
				input_loaders,
				schemas,
				table_names,
				std::vector<std::string>(
						query.begin() + other_depth_one_start,
						query.end()),
						queryContext,
						call_depth + 1
		);

		blazing_frame result_frame;
		if(ral::operators::is_join(query[0])){
			//we know that left and right are dataframes we want to join together
			left_frame.add_table(right_frame.get_columns()[0]);
			///left_frame.consolidate_tables();
			blazing_timer.reset();
			result_frame = ral::operators::process_join(queryContext, left_frame, query[0]);
			Library::Logging::Logger().logInfo("process_join took " + std::to_string(blazing_timer.getDuration()) + " ms with an output of " + std::to_string(result_frame.get_num_rows_in_table(0)));
			return result_frame;
		}else if(is_union(query[0])){
			//TODO: append the frames to each other
			//return right_frame;//!!
			blazing_timer.reset();
			result_frame = process_union(left_frame,right_frame,query[0]);
			Library::Logging::Logger().logInfo("process_union took " + std::to_string(blazing_timer.getDuration()) + " ms with an output of " + std::to_string(result_frame.get_num_rows_in_table(0)));
			return result_frame;
		}else{
			throw std::runtime_error{"In evaluate_split_query function: unsupported query operator"};
		}

	}else{
		//process child
		blazing_frame child_frame = evaluate_split_query(
				input_loaders,
				schemas,
				table_names,
				std::vector<std::string>(
						query.begin() + 1,
						query.end()),
						queryContext,
						call_depth + 1
		);
		//process self
		if(is_project(query[0])){
			blazing_timer.reset();
			execute_project_plan(child_frame,query[0]);
			Library::Logging::Logger().logInfo("process_project took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_num_rows_in_table(0)) + " rows");
			return child_frame;
		}else if(is_aggregate(query[0])){
			blazing_timer.reset();
			ral::operators::process_aggregate(child_frame, query[0], queryContext);
			Library::Logging::Logger().logInfo("process_aggregate took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_num_rows_in_table(0)) + " rows");
			return child_frame;
		}else if(is_sort(query[0])){
			blazing_timer.reset();
			ral::operators::process_sort(child_frame, query[0], queryContext);
			Library::Logging::Logger().logInfo("process_sort took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_num_rows_in_table(0)) + " rows");
			return child_frame;
		}else if(is_filter(query[0])){
			blazing_timer.reset();
			process_filter(child_frame,query[0]);
			Library::Logging::Logger().logInfo("process_filter took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_num_rows_in_table(0)) + " rows");
			return child_frame;
		}else{
			throw std::runtime_error{"In evaluate_split_query function: unsupported query operator"};
		}
		//return frame
	}
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
	blazing_frame output_frame = evaluate_split_query(input_tables, table_names, column_names, splitted, nullptr);

	for(size_t i=0;i<output_frame.get_width();i++){

		GDFRefCounter::getInstance()->deregister_column(output_frame.get_column(i).get_gdf_column());
		outputs.push_back(output_frame.get_column(i));

	}

	return GDF_SUCCESS;
}

gdf_error evaluate_query(
		std::vector<ral::io::data_loader > & input_loaders,
		std::vector<ral::io::Schema> & schemas,
		std::vector<std::string> table_names,
		std::string logicalPlan,
		std::vector<gdf_column_cpp> & outputs){

	std::vector<std::string> splitted = StringUtil::split(logicalPlan, "\n");
	if (splitted[splitted.size() - 1].length() == 0) {
		splitted.erase(splitted.end() -1);
	}

	blazing_frame output_frame = evaluate_split_query(input_loaders,schemas, table_names, splitted, nullptr);

	for(size_t i=0;i<output_frame.get_width();i++){

		GDFRefCounter::getInstance()->deregister_column(output_frame.get_column(i).get_gdf_column());
		outputs.push_back(output_frame.get_column(i));

	}

	return GDF_SUCCESS;
}

query_token_t evaluate_query(
		std::vector<ral::io::data_loader > input_loaders,
		std::vector<ral::io::Schema> schemas,
		std::vector<std::string> table_names,
		std::string logicalPlan,
		connection_id_t connection,
		const Context& queryContext,
		query_token_t token ){

	std::thread t([=]	{
		ral::config::GPUManager::getInstance().setDevice();

		std::vector<std::string> splitted = StringUtil::split(logicalPlan, "\n");
		if (splitted[splitted.size() - 1].length() == 0) {
			splitted.erase(splitted.end() -1);
		}

		try
		{
			CodeTimer blazing_timer;
			blazing_frame output_frame = evaluate_split_query(input_loaders, schemas,table_names, splitted, &queryContext);
			double duration = blazing_timer.getDuration();

			//REMOVE any columns that were ipcd to put into the result set
			for(size_t index = 0; index < output_frame.get_size_columns(); index++){
				gdf_column_cpp output_column = output_frame.get_column(index);

				if(output_column.is_ipc() || output_column.has_token()){
					output_frame.set_column(index,
							output_column.clone(output_column.name()));
				}
			}

			result_set_repository::get_instance().update_token(token, output_frame, duration);
		}
		catch(const std::exception& e)
		{
			std::cerr << "evaluate_split_query error => " << e.what() << '\n';
			try
			{
				result_set_repository::get_instance().update_token(token, blazing_frame{}, 0.0, e.what());
			}
			catch(const std::exception& e)
			{
				std::cerr << "error => " << e.what() << '\n';
			}
		}
	});

	//@todo: hablar con felipe sobre detach
	t.detach();

	return token;
}
