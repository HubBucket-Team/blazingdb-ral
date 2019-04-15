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
#include "operators/OrderBy.h"
#include "operators/GroupBy.h"
#include "operators/JoinOperator.h"

const std::string LOGICAL_UNION_TEXT = "LogicalUnion";
const std::string LOGICAL_SCAN_TEXT = "TableScan";
const std::string LOGICAL_PROJECT_TEXT = "LogicalProject";
const std::string LOGICAL_FILTER_TEXT = "LogicalFilter";

bool is_union(std::string query_part){
	return (query_part.find(LOGICAL_UNION_TEXT) != std::string::npos);
}

bool is_project(std::string query_part){
	return (query_part.find(LOGICAL_PROJECT_TEXT) != std::string::npos);
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

project_plan_params parse_project_plan(blazing_frame& input, std::string query_part) {

	gdf_error err = GDF_SUCCESS;

	// std::cout<<"starting process_project"<<std::endl;

	size_t size = input.get_column(0).size();


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

			add_expression_to_plan(	input,
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

	// std::cout<<"starting process_project"<<std::endl;

	size_t size = input.get_column(0).size();

	// LogicalProject(x=[$0], y=[$1], z=[$2], e=[$3], join_x=[$4], y0=[$5], EXPR$6=[+($0, $5)])
	std::string combined_expression = query_part.substr(
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
			output_type_expressions[i] = get_output_type_expression(&input, &max_temp_type, expression);
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

			evaluate_expression(input, expression,  output);

			columns[i] = output;
		}else{
			int index = get_index(expression);

			//if the column came over via ipc or was already used
			//we dont want to modify in place


			//			if(input_used_in_output[index] || input.get_column(index).is_ipc()){
			//becuase we already used this we can't just 0 copy it
			//we have to make a copy of it here

			gdf_column_cpp output = input.get_column(index).clone(name);
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

//TODO: this does not compact the allocations which would be nice if it could
void process_filter(blazing_frame & input, std::string query_part){
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
	gdf_dtype output_type = get_output_type_expression(&input, &max_temp_type, get_named_expression(query_part,"condition"));

	Library::Logging::Logger().logInfo("-> Filter sub block 2 took " + std::to_string(timer.getDuration()) + " ms");

	timer.reset();
	std::string conditional_expression = get_named_expression(query_part,"condition");
	Library::Logging::Logger().logInfo("-> Filter sub block 3 took " + std::to_string(timer.getDuration()) + " ms");
	// timer.reset();
	evaluate_expression(input, conditional_expression, stencil);

	// Library::Logging::Logger().logInfo("-> Filter sub block 4 took " + std::to_string(timer.getDuration()) + " ms");

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
	CUDF_CALL( gdf_apply_boolean_mask( index_col.get_gdf_column(), stencil.get_gdf_column(), temp_idx.get_gdf_column())	);
	Library::Logging::Logger().logInfo("-> Filter sub block 6 took " + std::to_string(timer.getDuration()) + " ms");

	timer.reset();
	gdf_column_cpp materialize_temp;
	materialize_temp.create_gdf_column(input.get_column(0).dtype(),temp_idx.size(),nullptr,get_width_dtype(max_temp_type), "");
	for(int i = 0; i < input.get_width();i++){
		materialize_temp.set_dtype(input.get_column(i).dtype());

		materialize_column(
				input.get_column(i).get_gdf_column(),
				materialize_temp.get_gdf_column(),
				temp_idx.get_gdf_column()
		);

		materialize_temp.update_null_count();
		input.set_column(i,materialize_temp.clone(input.get_column(i).name()));
	}
	Library::Logging::Logger().logInfo("-> Filter sub block 7 took " + std::to_string(timer.getDuration()) + " ms");

	//free_gdf_column(&stencil);
	//free_gdf_column(&temp);
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
			Library::Logging::Logger().logInfo("process_join took " + std::to_string(blazing_timer.getDuration()) + " ms with an output of " + std::to_string(result_frame.get_column(0).size()));
			return result_frame;
		}else if(is_union(query[0])){
			//TODO: append the frames to each other
			//return right_frame;//!!
			blazing_timer.reset();
			result_frame = process_union(left_frame,right_frame,query[0]);
			Library::Logging::Logger().logInfo("process_union took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(left_frame.get_column(0).size()) + " rows with an output of " + std::to_string(result_frame.get_column(0).size()));
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
			Library::Logging::Logger().logInfo("process_project took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_column(0).size()) + " rows");
			return child_frame;
		}else if(ral::operators::is_aggregate(query[0])){
			blazing_timer.reset();
			ral::operators::process_aggregate(child_frame, query[0], queryContext);
			Library::Logging::Logger().logInfo("process_aggregate took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_column(0).size()) + " rows");
			return child_frame;
		}else if(ral::operators::is_sort(query[0])){
			blazing_timer.reset();
			ral::operators::process_sort(child_frame, query[0], queryContext);
			Library::Logging::Logger().logInfo("process_sort took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_column(0).size()) + " rows");
			return child_frame;
		}else if(is_filter(query[0])){
			blazing_timer.reset();
			process_filter(child_frame,query[0]);
			Library::Logging::Logger().logInfo("process_filter took " + std::to_string(blazing_timer.getDuration()) + " ms for " + std::to_string(child_frame.get_column(0).size()) + " rows");
			return child_frame;
		}else{
			throw std::runtime_error{"In evaluate_split_query function: unsupported query operator"};
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
		std::vector<void *> handles,
		const Context& queryContext){
	//register the query so we can receive result requests for it
	query_token_t token = result_set_repository::get_instance().register_query(connection);

	std::thread t = std::thread([=]	{
		std::vector<std::string> splitted = StringUtil::split(logicalPlan, "\n");
		if (splitted[splitted.size() - 1].length() == 0) {
			splitted.erase(splitted.end() -1);
		}

		try
		{
			CodeTimer blazing_timer;
			blazing_frame output_frame = evaluate_split_query(input_tables, table_names, column_names, splitted, &queryContext);
			double duration = blazing_timer.getDuration();

			//REMOVE any columns that were ipcd to put into the result set
			std::set<gdf_column *> included_columns;
			for(size_t index = 0; index < output_frame.get_size_columns(); index++){
				gdf_column_cpp output_column = output_frame.get_column(index);
				output_frame.set_column(index, output_column.clone(output_column.name()));

				// WSM IS THIS CORRECT, THIS IS PRIOR TO MERGE NEED TO LOOK INTO THIS
				/*if(output_column.is_ipc() || included_columns.find(output_column.get_gdf_column()) != included_columns.end()){
				output_frame.set_column(index,
						output_column.clone(output_column.name()));
				}else{
					output_column.delete_set_name(output_column.name());
				}*/
			}

			result_set_repository::get_instance().update_token(token, output_frame, duration);
		}
		catch(const std::exception& e)
		{
			std::cerr << "evaluate_split_query error => " << e.what() << '\n';
			result_set_repository::get_instance().update_token(token, blazing_frame{}, 0.0, e.what());
		}

		//@todo: hablar con felipe sobre cudaIpcCloseMemHandle
		for(int i = 0; i < handles.size(); i++){
			cudaIpcCloseMemHandle(handles[i]);
		}
	});

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
	blazing_frame output_frame = evaluate_split_query(input_tables, table_names, column_names, splitted, nullptr);

	for(size_t i=0;i<output_frame.get_width();i++){

		GDFRefCounter::getInstance()->deregister_column(output_frame.get_column(i).get_gdf_column());
		outputs.push_back(output_frame.get_column(i));

	}

	return GDF_SUCCESS;
}
