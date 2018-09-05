#include "CalciteInterpreter.h"
#include "StringUtil.h"
#include "DataFrame.h"
#include <algorithm>
#include <thread>

#include "Utils.cuh"
#include "LogicalFilter.h"
#include "JoinProcessor.h"

const std::string LOGICAL_JOIN_TEXT = "LogicalJoin";
const std::string LOGICAL_UNION_TEXT = "LogicalUnion";
const std::string LOGICAL_SCAN_TEXT = "LogicalScan";
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
	gdf_column temp;
	size_t size = input.get_size_column();

	create_gdf_column(&temp,GDF_INT64,size,nullptr,8);

	// LogicalProject(x=[$0], y=[$1], z=[$2], e=[$3], join_x=[$4], y0=[$5], EXPR$6=[+($0, $5)])
	std::string combined_expression = query_part.substr(
			query_part.find("(") + 1,
			(query_part.rfind(")") - query_part.find("(")) - 1
	);


	std::vector<std::string> expressions = StringUtil::split(combined_expression,"], ");
	//now we have a vector
	//x=[$0
	std::vector<bool> input_used_in_output(size,false);

	std::vector<gdf_column * > columns(expressions.size());
	for(int i = 0; i < expressions.size(); i++){

		std::string expression = expressions[i].substr(
				expressions[i].find("=[") + 2 ,
				(expressions[i].size() - expressions[i].find("=[")) - 2
		);

		if(contains_evaluation(expression)){
			//assumes worst possible case allocation for output
			//TODO: find a way to know what our output size will be
			gdf_column * output = new gdf_column;
			create_gdf_column(output,output->dtype,size,nullptr,8);
			gdf_error err = evaluate_expression(
					input,
					expression,
					output,
					&temp);
			columns[i] = output;
		}else{
			int index = get_index(expression);
			columns[i] = input.get_column(index);
			if(input_used_in_output[index]){
				//becuase we already used this we can't just 0 copy it
				//we have to make a copy of it here
				gdf_column * output = new gdf_column;
				gdf_column empty;

				int width;
				get_column_byte_width(input.get_column(index), &width);
				create_gdf_column(&empty,input.get_column(index)->dtype,0,nullptr,width);
				create_gdf_column(output,input.get_column(index)->dtype,size,nullptr,width);
				//TODO: verify that this works concat whoudl be able to take in an empty one
				//even better would be if we could pass it in a  a null pointer and use it for copy
				gdf_error err = gpu_concat(input.get_column(index), &empty, output);

				free_gdf_column(&empty);
			}else{
				input_used_in_output[i] = true;
			}
		}
	}

	for(int i = 0; i < expressions.size(); i++){
		if(!input_used_in_output[i]){
			//free up the space
			free_gdf_column(input.get_column(i));

		}
	}


	input.clear();
	input.add_table(columns);

	free_gdf_column(&temp);

}

std::string get_named_expression(std::string query_part, std::string expression_name){
	int start_position =( query_part.find(expression_name + "=["))+ 2 + expression_name.length();
	int end_position = (query_part.find("]",start_position));
	return query_part.substr(start_position,end_position - start_position);
}


gdf_error process_join(blazing_frame & input, std::string query_part){

	size_t size = input.get_size_column();

	gdf_column left_indices, right_indices;
	//right now it outputs int32
	create_gdf_column(&left_indices,GDF_INT32,size,nullptr,sizeof(int));

	std::string condition = get_condition_expression(query_part);
	std::string join_type = get_named_expression(query_part,"joinType");

	gdf_error err = evaluate_join(
			condition,
			join_type,
			input,
			&left_indices,
			&right_indices
	);
}

gdf_error process_sort(blazing_frame & input, std::string query_part){


	/*gdf_error gdf_order_by(size_t nrows,     //in: # rows
		       gdf_column* cols, //in: host-side array of gdf_columns
		       size_t ncols,     //in: # cols
		       void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
		       int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
		       size_t* d_indx);*/

	std::string combined_expression = query_part.substr(
			query_part.find("("),
			(query_part.rfind(")") - query_part.find("(")) - 1
	);
	//LogicalSort(sort0=[$4], sort1=[$7], dir0=[ASC], dir1=[ASC])
	size_t num_sort_columns = count_string_occurrence(combined_expression,"sort");

	void** d_cols;
	cudaMalloc(d_cols,sizeof(void*) * input.get_width());
	int * d_types;
	cudaMalloc(d_cols,sizeof(int) * input.get_width());
	gdf_column * cols = new gdf_column[num_sort_columns];
	std::vector<size_t> sort_column_indices(num_sort_columns);

	for(int i = 0; i < num_sort_columns; i++){
		int sort_column_index = get_index(
				get_named_expression(
						combined_expression,
						"sort" + std::to_string(i)
				)
		);

		//TODO: get ascending or descending but right now thats not being used
		gdf_column * other_column = input.get_column(sort_column_index);
		cols[i].data = other_column->data;
		cols[i].dtype = other_column->dtype;
		cols[i].dtype_info = other_column->dtype_info;
		cols[i].null_count = other_column->null_count;
		cols[i].size = other_column->size;
		cols[i].valid = other_column->valid;
	}

	size_t * indices;
	cudaMalloc((void**)&indices,sizeof(size_t) * input.get_column(0)->size);
	gdf_error err = gdf_order_by(
			input.get_column(0)->size,
			cols,
			1, //?
			d_cols,
			d_types,
			indices
	);

	cudaFree(d_cols);
	cudaFree(d_types);

	int widest_column = 0;
	for(int i = 0; i < input.get_width();i++){
		int cur_width;
		get_column_byte_width(input.get_column(i), &cur_width);
		if(cur_width > widest_column){
			widest_column = cur_width;
		}

	}
	//find the widest possible column

	gdf_column temp_output;
	create_gdf_column(&temp_output,input.get_column(0)->dtype,input.get_column(0)->size,nullptr,widest_column);
	//now we need to materialize
	//i dont think we can do that in place since we are writing and reading out of order
	for(int i = 0; i < input.get_width();i++){
		temp_output.dtype = input.get_column(i)->dtype;
		/*gdf_error err = materialize_column_size_t(
				input.get_column(i),
				&temp_output,
				indices
		);*/

		gdf_column empty;

		int width;
		get_column_byte_width(input.get_column(i), &width);
		create_gdf_column(&empty,input.get_column(i)->dtype,0,nullptr,width);

		//copy output back to dat aframe
		err = gpu_concat(&temp_output, &empty, input.get_column(i));
		free_gdf_column(&empty);
	}
	//TODO: handle errors
	cudaFree(indices);
	delete[] cols;
	free_gdf_column(&temp_output);
	return GDF_SUCCESS;
}

//TODO: this does not compact the allocations which would be nice if it could
gdf_error process_filter(blazing_frame & input, std::string query_part){
	gdf_column stencil, temp;
	create_gdf_column(&stencil,GDF_INT8,input.get_column(0)->size,nullptr,1);
	create_gdf_column(&temp,GDF_INT64,input.get_column(0)->size,nullptr,8);


	gdf_error err = evaluate_expression(
			input,
			get_condition_expression(query_part),
			&stencil,
			&temp);

	if(err == GDF_SUCCESS){
		//apply filter to all the columns
		for(int i = 0; i < input.get_width(); i++){
			err = gpu_apply_stencil(
					input.get_column(i),
					&stencil,
					input.get_column(i)
			);
			if(err != GDF_SUCCESS){
				free_gdf_column(&stencil);
				free_gdf_column(&temp);
				return err;
			}
		}

	}else{
		free_gdf_column(&stencil);
		free_gdf_column(&temp);
		return err;
	}
	free_gdf_column(&stencil);
	free_gdf_column(&temp);
	return GDF_SUCCESS;

}

//Returns the index from table if exists
size_t get_table_index(std::vector<std::string> table_names, std::string table_name){
	auto it = std::find(table_names.begin(), table_names.end(), table_name);
	if(it != table_names.end()){
		return std::distance(table_names.begin(), it);
	}else{
		throw std::invalid_argument( "index does not exists" );
	}
}

//TODO: if a table needs to be used more than once you need to include it twice
//i know that kind of sucks, its for the 0 copy stuff, this can easily be remedied
//by changings scan to make copies
blazing_frame evaluate_split_query(
		std::vector<std::vector<gdf_column *> > input_tables,
		std::vector<std::string> table_names,
		std::vector<std::vector<std::string>> column_names,
		std::vector<std::string> query){
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
		//process left
		int other_depth_one_start = 2;
		for(int i = 2; i < query.size(); i++){
			int j = 0;
			while(query[i][j] == ' '){

				j+=2;
			}
			int depth = j / 2;
			if(depth == 1){
				other_depth_one_start = i;
			}
		}
		//these shoudl be split up and run on different threads
		blazing_frame left_frame;
		std::thread left_thread =
				std::thread([&left_frame, &input_tables,&table_names,&column_names,&query,other_depth_one_start](){
			left_frame = evaluate_split_query(
					input_tables,
					table_names,
					column_names,
					std::vector<std::string>(
							query.begin() + 1,
							query.begin() + other_depth_one_start)
			);

		});

		blazing_frame right_frame;
		std::thread right_thread =
				std::thread([&right_frame, &input_tables,&table_names,&column_names,&query,other_depth_one_start](){
			right_frame = evaluate_split_query(
					input_tables,
					table_names,
					column_names,
					std::vector<std::string>(
							query.begin() + other_depth_one_start,
							query.end())
			);

		});

		left_thread.join();
		right_thread.join();


		if(is_join(query[0])){
			//we know that left and right are dataframes we want to join together
			return left_frame;//!!
			//return process_join(left_frame,right_frame,query[0]);
		}else if(is_union(query[0])){
			return right_frame;//!!
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
						query.end())
		);
		//process self
		if(is_project(query[0])){
			gdf_error err = process_project(child_frame,query[0]);
			return child_frame;
		}else if(is_aggregate(query[0])){
			//gdf_error err = process_aggregate(child_frame,query[0]);
			return child_frame;
		}else if(is_sort(query[0])){
			gdf_error err = process_sort(child_frame,query[0]);
			return child_frame;
		}else if(is_filter(query[0]))
		{
			gdf_error err = process_filter(child_frame,query[0]);
			return child_frame;
		}else{
			//some error
		}
		//return frame
	}

	int max_depth = 0;
	int count_depth_1 = 0; //should always be 1 or 2

	//find max depth here

	int * node_ends = new int[count_depth_1 + 1];
	node_ends[0] = 0;
	//so any node who has a max depth of 1 gets processed by this

	if(max_depth == 1){
		//process element children
		//then self
	}else{
		//for each node where depth == 1
		//get that node and its children nodes
		//if I am not mistaken
		blazing_frame * new_frames[count_depth_1];
		for(int i = 0; i < count_depth_1; i++){
			*new_frames[i] = evaluate_split_query(
					input_tables,
					table_names,
					column_names,
					std::vector<std::string>(query.begin() + node_ends[i], query.begin() + node_ends[i + 1]));
		}
		//get me my chidlren and do them instead
		//then do myself
	}
}

gdf_error evaluate_query(
		std::vector<std::vector<gdf_column *> > input_tables,
		std::vector<std::string> table_names,
		std::vector<std::vector<std::string>> column_names,
		std::string query,
		std::vector<gdf_column *> & outputs,
		std::vector<std::string> & output_column_names,
		void * temp_space){

	std::vector<std::string> splitted = StringUtil::split(query, '\n');
	for(auto str : splitted)
		std::cout<<StringUtil::rtrim(str)<<"\n";
	blazing_frame output_frame = evaluate_split_query(input_tables, table_names, column_names, splitted);

	size_t cur_count = 0;
	for(size_t i=0;i<output_frame.get_width();i++){
		for(size_t j=0;j<output_frame.get_size_column(i);j++){
			outputs.push_back(output_frame.get_column(cur_count));
			cur_count++;
		}
	}
}
