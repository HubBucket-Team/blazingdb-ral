#include "CalciteInterpreter.h"
#include "StringUtil.h"

#include <algorithm>
#include <thread>
#include <regex>

#include "Utils.cuh"
#include "LogicalFilter.h"
#include "ResultSetRepository.h"
#include "JoinProcessor.h"
#include "ColumnManipulation.cuh"
#include "CalciteExpressionParsing.h"

const std::string LOGICAL_JOIN_TEXT = "LogicalJoin";
const std::string LOGICAL_UNION_TEXT = "LogicalUnion";
const std::string LOGICAL_SCAN_TEXT = "TableScan";
const std::string LOGICAL_AGGREGATE_TEXT = "LogicalAggregate";
const std::string LOGICAL_PROJECT_TEXT = "LogicalProject";
const std::string LOGICAL_SORT_TEXT = "LogicalSort";
const std::string LOGICAL_FILTER_TEXT = "LogicalFilter";


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
	return (expression.find("(") != std::string::npos);
}

gdf_error process_project(blazing_frame & input, std::string query_part){
	size_t size = input.get_column(0).size();


	// LogicalProject(x=[$0], y=[$1], z=[$2], e=[$3], join_x=[$4], y0=[$5], EXPR$6=[+($0, $5)])
	std::string combined_expression = query_part.substr(
			query_part.find("(") + 1,
			(query_part.rfind(")") - query_part.find("(")) - 1
	);

	std::regex pattern ("[\\w$_]+=\\[.*?\\]");
	std::smatch matcher;
	std::vector<std::string> expressions;

	while (std::regex_search (combined_expression, matcher, pattern)) {
		for (auto match:matcher)
			expressions.push_back(match);
		combined_expression = matcher.suffix().str();
	}

	//std::vector<std::string> expressions = StringUtil::split(combined_expression,"]");
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

	gdf_column_cpp temp;
	if(max_temp_type != GDF_invalid){
		temp.create_gdf_column(max_temp_type,size,nullptr,get_width_dtype(max_temp_type));
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
			output.create_gdf_column(output_type_expressions[i],size,nullptr,get_width_dtype(output_type_expressions[i]));

			gdf_error err = evaluate_expression(
					input,
					expression,
					output,
					temp);

			columns[i] = output;

			if(err != GDF_SUCCESS){
				//TODO: clean up everything here so we dont run out of memory
				return err;
			}
		}else{
			int index = get_index(expression);

			//if the column came over via ipc or was already used
			//we dont want to modify in place


			if(input_used_in_output[index] || columns[i].is_ipc()){
				//becuase we already used this we can't just 0 copy it
				//we have to make a copy of it here
				gdf_column_cpp output;
				gdf_column_cpp empty;

				int width;
				get_column_byte_width(input.get_column(index).get_gdf_column(), &width);

				empty.create_gdf_column(input.get_column(index).dtype(),0,nullptr,width);
				output.create_gdf_column(input.get_column(index).dtype(),size,nullptr,width);
				//TODO: verify that this works concat whoudl be able to take in an empty one
				//even better would be if we could pass it in a  a null pointer and use it for copy
				gdf_error err = gpu_concat(input.get_column(index).get_gdf_column(), empty.get_gdf_column(), output.get_gdf_column());
				if(err != GDF_SUCCESS){
					//TODO: clean up everything here so we dont run out of memory
					return err;
				}
				columns[i] = output;
			}else{
				input_used_in_output[index] = true;
				columns[i] = input.get_column(index);
			}
		}
	}


	input.clear();
	input.add_table(columns);

	//free_gdf_column(&temp);
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

	size_t size = 0; //libgdf will be handling the outputs for these

	gdf_column_cpp left_indices, right_indices;
	//right now it outputs int32
	left_indices.create_gdf_column(GDF_INT32,size,nullptr,sizeof(int));
	right_indices.create_gdf_column(GDF_INT32,size,nullptr,sizeof(int));

	std::string condition = get_condition_expression(query_part);
	std::string join_type = get_named_expression(query_part,"joinType");

	gdf_error err = evaluate_join(
			condition,
			join_type,
			input,
			left_indices.get_gdf_column(),
			right_indices.get_gdf_column()
	);

	size_t allocation_size_valid = ((((left_indices.get_gdf_column()->size + 7 ) / 8) + 63 ) / 64) * 64; //so allocations are supposed to be 64byte aligned

	cudaMalloc((void **) &left_indices.get_gdf_column()->valid, allocation_size_valid);
	cudaMalloc((void **) &right_indices.get_gdf_column()->valid, allocation_size_valid);
	cudaMemset(left_indices.get_gdf_column()->valid, (gdf_valid_type) 255, allocation_size_valid);
	cudaMemset(right_indices.get_gdf_column()->valid, (gdf_valid_type) 255, allocation_size_valid);

	if(err != GDF_SUCCESS){
		//TODO: clean up everything here so we dont run out of memory
		//return err;
	}
	//the options get interesting here. So if the join nis smaller than the input
	// you could write the output in place, saving time for allocations then shrink later on
	// the simplest solution is to reallocate space and free up the old after copying it over

	//a data frame should have two "tables"or groups of columns at this point
	std::vector<gdf_column_cpp> new_columns(input.get_size_columns());
	size_t first_table_end_index = input.get_size_column();
	int column_width;
	for(int column_index = 0; column_index < input.get_size_columns(); column_index++){
		gdf_column_cpp output;

		get_column_byte_width(input.get_column(column_index).get_gdf_column(), &column_width);
		output.create_gdf_column(input.get_column(column_index).dtype(),left_indices.size(),nullptr,column_width);

		if(column_index < first_table_end_index)
		{
			//materialize with left_indices
			err = materialize_column(input.get_column(column_index).get_gdf_column(),output.get_gdf_column(),left_indices.get_gdf_column());
		}else{
			//materialize with right indices
			err = materialize_column(input.get_column(column_index).get_gdf_column(),output.get_gdf_column(),right_indices.get_gdf_column());
		}
		if(err != GDF_SUCCESS){
			//TODO: clean up all the resources
			//return err;
		}
		//free_gdf_column(input.get_column(column_index));
		new_columns[column_index] = output;
	}
	input.clear();
	input.add_table(new_columns);
	return input;
}

std::vector<size_t> get_group_columns(std::string query_part){

	std::string temp_column_string = get_named_expression(query_part,"group");
	//now you have somethig like {0, 1}
	temp_column_string = temp_column_string.substr(1,temp_column_string.length() - 2);
	std::vector<std::string> column_numbers_string = StringUtil::split(temp_column_string,",");
	std::vector<size_t> group_columns(column_numbers_string.size());
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



	//get groups
	std::vector<size_t> group_columns = get_group_columns(query_part);

	//get aggregations
	std::vector<gdf_agg_op> aggregation_types;
	std::vector<std::string>  aggregation_input_expressions;

	bool expressionFound = true;

	size_t expression_index = 0;
	while(expressionFound){

		std::string expression = get_named_expression(query_part,"EXPR$" + std::to_string(expression_index));

		if("" == expression){
			expressionFound = false;
		}else{
			gdf_agg_op operation;
			gdf_error err = get_aggregation_operation(expression,&operation);
			aggregation_types.push_back(operation);
			aggregation_input_expressions.push_back(get_string_between_outer_parentheses(expression));
		}

		expression_index++;
	}

	gdf_column_cpp temp;
	gdf_dtype max_temp_type = GDF_invalid;
	std::vector<gdf_dtype> aggregation_input_types;

	size_t size = input.get_column(0).size();
	size_t aggregation_size = size;

	for(int i = 0; i < aggregation_types.size(); i++){
		if(contains_evaluation(aggregation_input_expressions[i])){

			gdf_error err = get_output_type_expression(&input, &aggregation_input_types[i], &max_temp_type, aggregation_input_expressions[i]);
			if(get_width_dtype(max_temp_type) < get_width_dtype(aggregation_input_types[i])){
				max_temp_type = aggregation_input_types[i];
				//by doing this we can now use the temp space as where we put our reductions then output our reductions right back into the input
				//so long as the input isnt an ipc one
			}
			//temp.create_gdf_column(GDF_INT64,size,nullptr,8);
			//break;
		}
	}
	temp.create_gdf_column(max_temp_type,size,nullptr,get_width_dtype(max_temp_type));

	gdf_column ** group_by_columns_ptr = new gdf_column *[group_columns.size()];
	gdf_column ** group_by_columns_ptr_out = new gdf_column *[group_columns.size()];

	std::vector<gdf_column_cpp> output_columns_group;
	std::vector<gdf_column_cpp> output_columns_aggregations;

	//create output here and pass in its pointers to this
	for(int group_columns_index = 0; group_columns_index < group_columns.size(); group_columns_index++){
		gdf_column_cpp input_column = input.get_column(group_columns[group_columns_index]);
		group_by_columns_ptr[group_columns_index] = input_column.get_gdf_column();
		gdf_column_cpp output_group;
		output_group.create_gdf_column(input_column.dtype(),size,nullptr,get_width_dtype(input_column.dtype()));
		output_columns_group.push_back(output_group);
		group_by_columns_ptr_out[group_columns_index] = output_group.get_gdf_column();
	}


	for(int i = 0; i < aggregation_types.size(); i++){
		std::string expression = aggregation_input_expressions[i];
		gdf_column_cpp aggregation_input;
		if(contains_evaluation(expression)){

			//we dont knwo what the size of this input will be so allcoate max size
			aggregation_input.create_gdf_column(aggregation_input_types[i],size,nullptr,get_width_dtype(aggregation_input_types[i]));

			gdf_error err = evaluate_expression(
					input,
					expression,
					aggregation_input,
					temp);

			if(err != GDF_SUCCESS){
				//TODO: clean up everything here so we dont run out of memory
				return err;
			}
		}else{
			aggregation_input = input.get_column(get_index(expression));
		}


		gdf_error err;
		gdf_dtype output_type = get_aggregation_output_type(aggregation_input.dtype(),aggregation_types[i]);
		gdf_column_cpp output_column;
		output_column.create_gdf_column(output_type,aggregation_size,nullptr,get_width_dtype(output_type));
		output_columns_aggregations.push_back(output_column);

		if(i > 0){
			//only want to output this data on the first iteration
			group_by_columns_ptr_out = nullptr;
		}

		gdf_context ctxt;
		ctxt.flag_distinct = aggregation_types[i] == GDF_COUNT_DISTINCT ? true : false;

		switch(aggregation_types[i]){
		case GDF_SUM:
			err = gdf_group_by_sum(group_columns.size(),group_by_columns_ptr,aggregation_input.get_gdf_column(),
					nullptr,group_by_columns_ptr_out,output_column.get_gdf_column(),&ctxt);
			if(err == GDF_SUCCESS){
				aggregation_size = output_column.size();
				//so that subsequent iterations won't be too large
			}else{
				//be just as responsible as all the other times we didn't do anything when we got an error bback!
			}
			break;
		case GDF_MIN:
			err = gdf_group_by_min(group_columns.size(),group_by_columns_ptr,aggregation_input.get_gdf_column(),
					nullptr,group_by_columns_ptr_out,output_column.get_gdf_column(),&ctxt);
			if(err == GDF_SUCCESS){
				aggregation_size = output_column.size();
				//so that subsequent iterations won't be too large
			}else{
				//be just as responsible as all the other times we didn't do anything when we got an error bback!
			}
			break;
		case GDF_MAX:
			err = gdf_group_by_max(group_columns.size(),group_by_columns_ptr,aggregation_input.get_gdf_column(),
					nullptr,group_by_columns_ptr_out,output_column.get_gdf_column(),&ctxt);
			if(err == GDF_SUCCESS){
				aggregation_size = output_column.size();
				//so that subsequent iterations won't be too large
			}else{
				//be just as responsible as all the other times we didn't do anything when we got an error bback!
			}
			break;
		case GDF_AVG:
			err = gdf_group_by_avg(group_columns.size(),group_by_columns_ptr,aggregation_input.get_gdf_column(),
					nullptr,group_by_columns_ptr_out,output_column.get_gdf_column(),&ctxt);
			if(err == GDF_SUCCESS){
				aggregation_size = output_column.size();
				//so that subsequent iterations won't be too large
			}else{
				//be just as responsible as all the other times we didn't do anything when we got an error bback!
			}
			break;
		case GDF_COUNT:
		case GDF_COUNT_DISTINCT:
			err = gdf_group_by_count(group_columns.size(),group_by_columns_ptr,aggregation_input.get_gdf_column(),
					nullptr,group_by_columns_ptr_out,output_column.get_gdf_column(),&ctxt);
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
		output_columns_aggregations[i].compact();
	}

	for(int i = 0; i < output_columns_group.size(); i++){
		output_columns_group[i].compact();
	}

	input.clear();
	input.add_table(output_columns_group);
	input.add_table(output_columns_aggregations);
	input.consolidate_tables();

}

gdf_error gdf_group_by_sum(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt);           //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct



gdf_error process_sort(blazing_frame & input, std::string query_part){


	/*gdf_error gdf_order_by(size_t nrows,     //in: # rows
		       gdf_column* cols, //in: host-side array of gdf_columns
		       size_t ncols,     //in: # cols
		       void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
		       int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
		       size_t* d_indx);*/
	std::cout<<"about to process sort"<<std::endl;
	std::string combined_expression = query_part.substr(
			query_part.find("("),
			(query_part.rfind(")") - query_part.find("(")) - 1
	);
	//LogicalSort(sort0=[$4], sort1=[$7], dir0=[ASC], dir1=[ASC])
	size_t num_sort_columns = count_string_occurrence(combined_expression,"sort");

	void** d_cols;

	std::vector<gdf_column_cpp> output_columns;


	cudaMalloc((void **) &d_cols,sizeof(void*) * num_sort_columns);
	int * d_types;
	cudaMalloc((void **)&d_types,sizeof(int) * num_sort_columns);
	gdf_column * cols = new gdf_column[num_sort_columns];
	std::vector<size_t> sort_column_indices(num_sort_columns);

	for(int i = 0; i < num_sort_columns; i++){
		int sort_column_index = get_index(
				get_named_expression(
						combined_expression,
						"sort" + std::to_string(i)
				)
		);

		cols[i] = *input.get_column(sort_column_index).get_gdf_column();
		//TODO: get ascending or descending but right now thats not being used
		/*
		gdf_column_cpp other_column = input.get_column(sort_column_index);
		cols[i].data = input.get_column(sort_column_index).data();
		cols[i].dtype = other_column.dtype();
		cols[i].dtype_info = other_column.dtype_info();
		cols[i].null_count = other_column.null_count();
		cols[i].size = other_column.size();
		cols[i].valid = other_column.valid();*/
	}

	size_t * indices;
	cudaMalloc((void **)&indices,sizeof(size_t) * input.get_column(0).size());
	gdf_error err = gdf_order_by(
			input.get_column(0).size(),
			cols,
			num_sort_columns,
			d_cols,
			d_types,
			indices
	);

	cudaFree(d_cols);
	cudaFree(d_types);

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
	temp_output.create_gdf_column(input.get_column(0).dtype(),input.get_column(0).size(),nullptr,widest_column);
	//now we need to materialize
	//i dont think we can do that in place since we are writing and reading out of order
	for(int i = 0; i < input.get_width();i++){
		temp_output.set_dtype(input.get_column(i).dtype());


		gdf_column_cpp indices_column((void *)indices,nullptr,GDF_UINT64,input.get_column(i).size(),0);

		gdf_error err = materialize_column(
				input.get_column(i).get_gdf_column(),
				temp_output.get_gdf_column(),
				indices_column.get_gdf_column()
		);

		gdf_column_cpp empty;

		int width;
		get_column_byte_width(input.get_column(i).get_gdf_column(), &width);
		empty.create_gdf_column(input.get_column(i).dtype(),0,nullptr,width);

		//copy output back to dat aframe

		gdf_column_cpp new_output;
		if(input.get_column(i).is_ipc()){
			new_output.create_gdf_column(input.get_column(i).dtype(), input.get_column(i).size(),nullptr,get_width_dtype(input.get_column(i).dtype()));
			input.set_column(i,new_output);
		}else{
			new_output = input.get_column(i);
		}
		err = gpu_concat(temp_output.get_gdf_column(), empty.get_gdf_column(), new_output.get_gdf_column());

		//free_gdf_column(&empty);
	}
	//TODO: handle errors
	cudaFree(indices);
	delete[] cols;
	//free_gdf_column(&temp_output);
	return GDF_SUCCESS;
}

//TODO: this does not compact the allocations which would be nice if it could
gdf_error process_filter(blazing_frame & input, std::string query_part){

	//assert(input.get_column(0) != nullptr);

	gdf_column_cpp stencil, temp;

	size_t size = input.get_column(0).size();

	stencil.create_gdf_column(GDF_INT8,input.get_column(0).size(),nullptr,1);

	gdf_dtype output_type_junk; //just gets thrown away
	gdf_dtype max_temp_type = GDF_INT8;
	for(int i = 0; i < input.get_width(); i++){
		if(get_width_dtype(input.get_column(i).dtype()) > get_width_dtype(max_temp_type)){
			max_temp_type = input.get_column(i).dtype();
		}
	}
	gdf_dtype output_type; // this is junk since we know the output types here
	gdf_error err = get_output_type_expression(&input, &output_type, &max_temp_type, get_condition_expression(query_part));
	if(err != GDF_SUCCESS){
		//panic then do wonderful things here to fix everything
		//im really liking Andrescus talk on control flow blah blah something i forget his name
	}

	temp.create_gdf_column(max_temp_type,input.get_column(0).size(),nullptr,get_width_dtype(max_temp_type));

	err = evaluate_expression(
			input,
			get_condition_expression(query_part),
			stencil,
			temp);

	if(err == GDF_SUCCESS){
		//apply filter to all the columns
		for(int i = 0; i < input.get_width(); i++){
			temp.set_dtype(input.get_column(i).dtype());
			//			cudaPointerAttributes attributes;
			//			cudaError_t err2 = cudaPointerGetAttributes ( &attributes, (void *) temp.data );
			//			err2 = cudaPointerGetAttributes ( &attributes, (void *) input.get_column(i)->data );
			//			err2 = cudaPointerGetAttributes ( &attributes, (void *) stencil.data );


			//just for testing
			//			cudaMalloc((void **)&(temp.data),1000);
			//			cudaMalloc((void **)&(temp.valid),1000);

			err = gpu_apply_stencil(
					input.get_column(i).get_gdf_column(),
					stencil.get_gdf_column(),
					temp.get_gdf_column()
			);




			//
			gdf_column_cpp new_output;
			gdf_column_cpp empty;

			int width;
			get_column_byte_width(input.get_column(i).get_gdf_column(), &width);
			empty.create_gdf_column(input.get_column(i).dtype(),0,nullptr,width);


			if(input.get_column(i).is_ipc()){
				//make a copy

				new_output.create_gdf_column(input.get_column(i).dtype(),input.get_column(i).size(),nullptr,get_width_dtype(input.get_column(i).dtype()));


				if(err != GDF_SUCCESS){
					return err;
				}


			}else{


				input.get_column(i).realloc_gdf_column(input.get_column(i).dtype(),temp.size(),width);
				new_output = input.get_column(i);

			}

			gdf_error err = gpu_concat(temp.get_gdf_column(), empty.get_gdf_column(), new_output.get_gdf_column());

			if(err != GDF_SUCCESS){
				return err;
			}
		}

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

		if(is_join(query[0])){
			//we know that left and right are dataframes we want to join together
			left_frame.add_table(right_frame.get_columns()[0]);
			///left_frame.consolidate_tables();
			return process_join(left_frame,query[0]);
		}else if(is_union(query[0])){
			//TODO: append the frames to each other
			//return right_frame;//!!
			//return process_union(left_frame,right_frame,query[0]);
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
			gdf_error err = process_project(child_frame,query[0]);
			return child_frame;
		}else if(is_aggregate(query[0])){
			gdf_error err = process_aggregate(child_frame,query[0]);
			return child_frame;
		}else if(is_sort(query[0])){
			gdf_error err = process_sort(child_frame,query[0]);
			return child_frame;
		}else if(is_filter(query[0])){
			gdf_error err = process_filter(child_frame,query[0]);
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
	print_column<int8_t>(input_tables[0][0].get_gdf_column());

	query_token_t token = result_set_repository::get_instance().register_query(connection); //register the query so we can receive result requests for it

	std::thread t = std::thread([&handles,logicalPlan, &input_tables, table_names, column_names,token]{
		std::vector<std::string> splitted = StringUtil::split(logicalPlan, "\n");
		blazing_frame output_frame = evaluate_split_query(input_tables, table_names, column_names, splitted);
		result_set_repository::get_instance().update_token(token, output_frame);

		for(int i = 0; i < handles.size(); i++){
			cudaIpcCloseMemHandle (handles[i]);
		}
		//			std::cout<<"Result\n";
		//			print_column<int8_t>(output_frame.get_columns()[0][0].get_gdf_column());
	});;

	t.detach();

	return token;
}

gdf_error evaluate_query(
		std::vector<std::vector<gdf_column_cpp> > input_tables,
		std::vector<std::string> table_names,
		std::vector<std::vector<std::string>> column_names,
		std::string logicalPlan,
		std::vector<gdf_column_cpp> & outputs){

	std::vector<std::string> splitted = StringUtil::split(logicalPlan, '\n');
	blazing_frame output_frame = evaluate_split_query(input_tables, table_names, column_names, splitted);

	for(size_t i=0;i<output_frame.get_width();i++){

		GDFRefCounter::getInstance()->deregister_column(output_frame.get_column(i).get_gdf_column());
		outputs.push_back(output_frame.get_column(i));

	}

	return GDF_SUCCESS;
}
