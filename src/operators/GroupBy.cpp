#include <iostream>
#include <future>
#include <functional>
#include <iterator>
#include <regex>
#include <blazingdb/io/Library/Logging/Logger.h>
#include <blazingdb/io/Util/StringUtil.h>
#include "GroupBy.h"
#include "CodeTimer.h"
#include "CalciteExpressionParsing.h"
#include "distribution/primitives.h"
#include "communication/CommunicationData.h"
#include "ColumnManipulation.cuh"
#include "GDFColumn.cuh"
#include "LogicalFilter.h"
#include "Traits/RuntimeTraits.h"
#include "groupby.hpp"
#include "table.hpp"
#include "reduction.hpp"

namespace ral {
namespace operators {

namespace {
using blazingdb::communication::Context;
} // namespace

const std::string LOGICAL_AGGREGATE_TEXT = "LogicalAggregate";

bool is_aggregate(std::string query_part){
	return (query_part.find(LOGICAL_AGGREGATE_TEXT) != std::string::npos);
}

std::vector<int> get_group_columns(std::string query_part){
	std::string temp_column_string = get_named_expression(query_part, "group");
	if(temp_column_string.size() <= 2){
		return std::vector<int>();
	}

	// Now we have somethig like {0, 1}
	temp_column_string = temp_column_string.substr(1, temp_column_string.length() - 2);
	std::vector<std::string> column_numbers_string = StringUtil::split(temp_column_string, ",");
	std::vector<int> group_column_indices(column_numbers_string.size());
	for(int i = 0; i < column_numbers_string.size();i++){
		group_column_indices[i] = std::stoull(column_numbers_string[i], 0);
	}
	return group_column_indices;
}

std::vector<gdf_column_cpp> groupby_without_aggregations(const std::vector<gdf_column_cpp> & input, const std::vector<int>& group_column_indices){

	gdf_size_type num_group_columns = group_column_indices.size();
	std::vector<gdf_column*> data_cols_in(input.size());
	for(int i = 0; i < input.size(); i++){
		data_cols_in[i] = input[i].get_gdf_column();
	}

	gdf_context ctxt;
	ctxt.flag_null_sort_behavior = GDF_NULL_AS_LARGEST; //  Nulls are are treated as largest
	ctxt.flag_groupby_include_nulls = 1; // Nulls are treated as values in group by keys where NULL == NULL (SQL style)

	cudf::table group_by_data_in_table(data_cols_in);
	cudf::table group_by_columns_out_table;
	rmm::device_vector<gdf_index_type> indexes_out;
	std::tie(group_by_columns_out_table, indexes_out) = gdf_group_by_without_aggregations(group_by_data_in_table, 
															num_group_columns, group_column_indices.data(), &ctxt);

	std::vector<gdf_column_cpp> output_columns_group(group_by_columns_out_table.num_columns());
	for(int i = 0; i < output_columns_group.size(); i++){
		group_by_columns_out_table.get_column(i)->col_name = nullptr; // need to do this because gdf_group_by_without_aggregations is not setting the name properly
		output_columns_group[i].create_gdf_column(group_by_columns_out_table.get_column(i));
	}
	gdf_column index_col;
	gdf_column_view(&index_col, static_cast<void*>(indexes_out.data().get()), nullptr,indexes_out.size(), GDF_INT32);

	std::vector<gdf_column_cpp> grouped_output(num_group_columns);
	for(int i = 0; i < num_group_columns; i++){
		if (input[i].valid())
			grouped_output[i].create_gdf_column(input[i].dtype(), index_col.size, nullptr, get_width_dtype(input[i].dtype()), input[i].name());
		else
			grouped_output[i].create_gdf_column(input[i].dtype(), index_col.size, nullptr, nullptr, get_width_dtype(input[i].dtype()), input[i].name());

		materialize_column(output_columns_group[i].get_gdf_column(),
											grouped_output[i].get_gdf_column(),
											&index_col);
	}
	return grouped_output;
}

void single_node_groupby_without_aggregations(blazing_frame& input, std::vector<int>& group_column_indices){

	std::vector<gdf_column_cpp> data_cols_in(input.get_width());
	for(int i = 0; i < input.get_width(); i++){
		data_cols_in[i] = input.get_column(i);
	}
	std::vector<gdf_column_cpp> grouped_table = groupby_without_aggregations(data_cols_in, group_column_indices);

	input.clear();
	input.add_table(grouped_table);
}

void distributed_groupby_without_aggregations(const Context& queryContext, blazing_frame& input, std::vector<int>& group_column_indices){
	using ral::communication::CommunicationData;

	std::vector<gdf_column_cpp> group_columns(group_column_indices.size());
	for(size_t i = 0; i < group_column_indices.size(); i++){
		group_columns[i] = input.get_column(group_column_indices[i]);
	}
	std::vector<gdf_column_cpp> data_cols_in(input.get_width());
	for(int i = 0; i < input.get_width(); i++){
		data_cols_in[i] = input.get_column(i);
	}

	size_t rowSize = input.get_num_rows_in_table(0);

	std::vector<gdf_column_cpp> selfSamples = ral::distribution::sampling::generateSample(group_columns, 0.1);

	auto groupByTask = std::async(std::launch::async,
																groupby_without_aggregations,
																std::ref(data_cols_in),
																std::ref(group_column_indices));

	std::vector<gdf_column_cpp> partitionPlan;
	if (queryContext.isMasterNode(CommunicationData::getInstance().getSelfNode())) {
		std::vector<ral::distribution::NodeSamples> samples = ral::distribution::collectSamples(queryContext);
    samples.emplace_back(rowSize, CommunicationData::getInstance().getSelfNode(), std::move(selfSamples));

		partitionPlan = ral::distribution::generatePartitionPlansGroupBy(queryContext, samples);

    ral::distribution::distributePartitionPlan(queryContext, partitionPlan);
	}
	else {
		ral::distribution::sendSamplesToMaster(queryContext, std::move(selfSamples), rowSize);

		partitionPlan = ral::distribution::getPartitionPlan(queryContext);
	}

	// Wait for groupByThread
	std::vector<gdf_column_cpp> groupedTable = groupByTask.get();

	std::vector<ral::distribution::NodeColumns> partitions = ral::distribution::partitionData(queryContext, groupedTable, group_column_indices, partitionPlan);

	ral::distribution::distributePartitions(queryContext, partitions);

	std::vector<ral::distribution::NodeColumns> partitionsToMerge = ral::distribution::collectPartitions(queryContext);
	auto it = std::find_if(partitions.begin(), partitions.end(), [&](ral::distribution::NodeColumns& el) {
			return el.getNode() == CommunicationData::getInstance().getSelfNode();
		});
	// Could "it" iterator be partitions.end()?
	partitionsToMerge.push_back(std::move(*it));

	ral::distribution::groupByWithoutAggregationsMerger(partitionsToMerge, group_column_indices, input);
}

void aggregations_with_groupby(gdf_agg_op agg_op, std::vector<gdf_column*>& group_by_columns_ptr, gdf_column_cpp& aggregation_input, std::vector<gdf_column*>& group_by_columns_ptr_out, gdf_column_cpp& output_column){
	gdf_context ctxt;
	ctxt.flag_distinct = (agg_op == GDF_COUNT_DISTINCT);
	ctxt.flag_method = GDF_HASH;
	ctxt.flag_sort_result = 1;

	switch(agg_op){
		case GDF_SUM:
			CUDF_CALL(gdf_group_by_sum(group_by_columns_ptr.size(),
																group_by_columns_ptr.data(),
																aggregation_input.get_gdf_column(),
																nullptr,
																group_by_columns_ptr_out.data(),
																output_column.get_gdf_column(),
																&ctxt));
			break;
		case GDF_MIN:
			CUDF_CALL(gdf_group_by_min(group_by_columns_ptr.size(),
																group_by_columns_ptr.data(),
																aggregation_input.get_gdf_column(),
																nullptr,
																group_by_columns_ptr_out.data(),
																output_column.get_gdf_column(),
																&ctxt));
			break;
		case GDF_MAX:
			CUDF_CALL(gdf_group_by_max(group_by_columns_ptr.size(),
																group_by_columns_ptr.data(),
																aggregation_input.get_gdf_column(),
																nullptr,
																group_by_columns_ptr_out.data(),
																output_column.get_gdf_column(),
																&ctxt));
			break;
		case GDF_AVG:
      CUDF_CALL(gdf_group_by_avg(group_by_columns_ptr.size(),
																group_by_columns_ptr.data(),
																aggregation_input.get_gdf_column(),
																nullptr,
																group_by_columns_ptr_out.data(),
																output_column.get_gdf_column(),
																&ctxt));
			break;
		case GDF_COUNT:
			CUDF_CALL(gdf_group_by_count(group_by_columns_ptr.size(),
																	group_by_columns_ptr.data(),
																	aggregation_input.get_gdf_column(),
																	nullptr,
																	group_by_columns_ptr_out.data(),
																	output_column.get_gdf_column(),
																	&ctxt));
			break;
		case GDF_COUNT_DISTINCT:
			CUDF_CALL(gdf_group_by_count_distinct(group_by_columns_ptr.size(),
																						group_by_columns_ptr.data(),
																						aggregation_input.get_gdf_column(),
																						nullptr,
																						group_by_columns_ptr_out.data(),
																						output_column.get_gdf_column(),
																						&ctxt));
			break;
		}
}

gdf_reduction_op gdf_agg_op_to_gdf_reduction_op(gdf_agg_op agg_op){
	switch(agg_op){
		case GDF_SUM:
			return GDF_REDUCTION_SUM;
		case GDF_MIN:
			return GDF_REDUCTION_MIN;
		case GDF_MAX:
			return GDF_REDUCTION_MAX; 
		default:
			std::cout<<"ERROR:	Unexpected gdf_agg_op"<<std::endl;
			return GDF_REDUCTION_SUM;
	}
}

void aggregations_without_groupby(gdf_agg_op agg_op, gdf_column_cpp& aggregation_input, gdf_column_cpp& output_column, gdf_dtype output_type, std::string output_column_name){
	
	gdf_column_cpp temp;
	switch(agg_op){
		case GDF_SUM:
		case GDF_MIN:
		case GDF_MAX:
			if (aggregation_input.size() == 0) {
				// Set output_column data to invalid
				gdf_scalar null_value;
				null_value.is_valid = false;
				null_value.dtype = output_type;
				output_column.create_gdf_column(null_value, output_column_name);	
				break;
			} else {
				gdf_reduction_op reduction_op = gdf_agg_op_to_gdf_reduction_op(agg_op);
				gdf_scalar reduction_out = cudf::reduction(aggregation_input.get_gdf_column(), reduction_op, output_type);
				output_column.create_gdf_column(reduction_out, output_column_name);
				break;
			}
		case GDF_AVG:
			if (aggregation_input.size() == 0 || (aggregation_input.size() == aggregation_input.null_count())) {
				// Set output_column data to invalid
				gdf_scalar null_value;
				null_value.is_valid = false;
				null_value.dtype = output_type;
				output_column.create_gdf_column(null_value, output_column_name);	
				break;
			} else {
				gdf_dtype sum_output_type = get_aggregation_output_type(aggregation_input.dtype(),GDF_SUM, false);
				gdf_scalar avg_sum_scalar = cudf::reduction(aggregation_input.get_gdf_column(), GDF_REDUCTION_SUM, sum_output_type);
				long avg_count = aggregation_input.get_gdf_column()->size - aggregation_input.get_gdf_column()->null_count;

				assert(output_type == GDF_FLOAT64);
				assert(sum_output_type == GDF_INT64 || sum_output_type == GDF_FLOAT64);
				
				gdf_scalar avg_scalar;
				avg_scalar.dtype = GDF_FLOAT64;
				avg_scalar.is_valid = true;
				if (avg_sum_scalar.dtype == GDF_INT64)
					avg_scalar.data.fp64 = (double)avg_sum_scalar.data.si64/(double)avg_count;
				else
					avg_scalar.data.fp64 = (double)avg_sum_scalar.data.fp64/(double)avg_count;

				output_column.create_gdf_column(avg_scalar, output_column_name);
				break;
			}			
		case GDF_COUNT:
		{
			gdf_scalar reduction_out;
			reduction_out.dtype = GDF_INT64;
			reduction_out.is_valid = true;
			reduction_out.data.si64 = aggregation_input.get_gdf_column()->size - aggregation_input.get_gdf_column()->null_count;   
			
			output_column.create_gdf_column(reduction_out, output_column_name);
			break;
		}
		case GDF_COUNT_DISTINCT:
		{
			// TODO not currently supported
			std::cout<<"ERROR: COUNT DISTINCT currently not supported without a group by"<<std::endl;
		}
	}
}

std::vector<gdf_column_cpp> compute_aggregations(blazing_frame& input, std::vector<int>& group_column_indices, std::vector<gdf_agg_op>& aggregation_types, std::vector<std::string>& aggregation_input_expressions, std::vector<std::string>& aggregation_column_assigned_aliases){
	size_t row_size = input.get_num_rows_in_table(0);

	std::vector<gdf_column*> group_by_columns_ptr(group_column_indices.size());
	std::vector<gdf_column_cpp> output_columns_group(group_column_indices.size());
	std::vector<gdf_column*> group_by_columns_ptr_out(group_column_indices.size());
	for(size_t i = 0; i < group_column_indices.size(); i++){
		//TODO: fix this input_column goes out of scope before its used
		//create output here and pass in its pointers to this
		gdf_column_cpp& input_column = input.get_column(group_column_indices[i]);

		group_by_columns_ptr[i] = input_column.get_gdf_column();

		output_columns_group[i].create_gdf_column(input_column.dtype(), row_size, nullptr, get_width_dtype(input_column.dtype()), input_column.name());
		group_by_columns_ptr_out[i] = output_columns_group[i].get_gdf_column();
	}

	// If we have no groups you will output only one row
	size_t aggregation_size = (group_column_indices.size() == 0 ? 1 : row_size);

	std::vector<gdf_column_cpp> output_columns_aggregations(aggregation_types.size());
	for(size_t i = 0; i < aggregation_types.size(); i++){
		std::string expression = aggregation_input_expressions[i];
		gdf_column_cpp aggregation_input;
		if(contains_evaluation(expression)){
			//we dont knwo what the size of this input will be so allcoate max size
			//TODO de donde saco el nombre de la columna aqui???
			gdf_dtype unused;
			gdf_dtype agg_input_type = get_output_type_expression(&input, &unused, expression);
			aggregation_input.create_gdf_column(agg_input_type, row_size, nullptr, get_width_dtype(agg_input_type), "");
			evaluate_expression(input, expression, aggregation_input);
		}else{
			aggregation_input = input.get_column(get_index(expression));
		}

		gdf_dtype output_type = get_aggregation_output_type(aggregation_input.dtype(), aggregation_types[i], group_column_indices.size() != 0);

		// if the aggregation was given an alias lets use it, otherwise we'll name it based on the aggregation and input
		std::string output_column_name = (aggregation_column_assigned_aliases[i] == ""
																			? (aggregator_to_string(aggregation_types[i]) + "(" + aggregation_input.name() + ")")
																			: aggregation_column_assigned_aliases[i]);
		
		if (group_column_indices.size() == 0) {
			aggregations_without_groupby(aggregation_types[i], aggregation_input, output_columns_aggregations[i], output_type, output_column_name);
		}else{
			output_columns_aggregations[i].create_gdf_column(output_type, aggregation_size, nullptr, get_width_dtype(output_type), output_column_name);
			aggregations_with_groupby(aggregation_types[i],
																group_by_columns_ptr,
																aggregation_input,
																group_by_columns_ptr_out,
																output_columns_aggregations[i]);
		}

		//so that subsequent iterations won't be too large
		aggregation_size = output_columns_aggregations[i].size();
	}

	//TODO: this is pretty crappy because its recalcluating the groups each time, this is becuase the libgdf api can
	//only process one aggregate at a time while it calculates the group,
	//these steps would have to be divided up in order to really work

	//TODO: consider compacting columns here before moving on
	// for(size_t i = 0; i < output_columns_aggregations.size(); i++){
	// 	output_columns_aggregations[i].resize(aggregation_size);
	// 	output_columns_aggregations[i].update_null_count();
	// }

	// for(size_t i = 0; i < output_columns_group.size(); i++){
	// 	output_columns_group[i].resize(aggregation_size);
	// 	output_columns_group[i].update_null_count();
	// }

	// Concat grouped columns and then aggregated columns
	std::vector<gdf_column_cpp> aggregatedTable(std::move(output_columns_group));
	aggregatedTable.insert(
		aggregatedTable.end(),
		std::make_move_iterator(output_columns_aggregations.begin()),
		std::make_move_iterator(output_columns_aggregations.end())
	);

	return aggregatedTable;
}

void single_node_aggregations(blazing_frame& input, std::vector<int>& group_column_indices, std::vector<gdf_agg_op>& aggregation_types, std::vector<std::string>& aggregation_input_expressions, std::vector<std::string>& aggregation_column_assigned_aliases) {
	std::vector<gdf_column_cpp> aggregatedTable = compute_aggregations(input,
																																		group_column_indices,
																																		aggregation_types,
																																		aggregation_input_expressions,
																																		aggregation_column_assigned_aliases);

	input.clear();
	input.add_table(aggregatedTable);
}

void distributed_aggregations_with_groupby(const Context& queryContext, blazing_frame& input, std::vector<int>& group_column_indices, std::vector<gdf_agg_op>& aggregation_types, std::vector<std::string>& aggregation_input_expressions, std::vector<std::string>& aggregation_column_assigned_aliases) {
	using ral::communication::CommunicationData;

	std::vector<gdf_column_cpp> group_columns(group_column_indices.size());
	for(size_t i = 0; i < group_column_indices.size(); i++){
		group_columns[i] = input.get_column(group_column_indices[i]);
	}

	size_t rowSize = input.get_num_rows_in_table(0);

	std::vector<gdf_column_cpp> selfSamples = ral::distribution::sampling::generateSample(group_columns, 0.1);

	auto aggregationTask = std::async(std::launch::async,
																		compute_aggregations,
																		std::ref(input),
																		std::ref(group_column_indices),
																		std::ref(aggregation_types),
																		std::ref(aggregation_input_expressions),
																		std::ref(aggregation_column_assigned_aliases));

	std::vector<gdf_column_cpp> partitionPlan;
	if (queryContext.isMasterNode(CommunicationData::getInstance().getSelfNode())) {
		std::vector<ral::distribution::NodeSamples> samples = ral::distribution::collectSamples(queryContext);
    samples.emplace_back(rowSize, CommunicationData::getInstance().getSelfNode(), std::move(selfSamples));

		partitionPlan = ral::distribution::generatePartitionPlansGroupBy(queryContext, samples);

    ral::distribution::distributePartitionPlan(queryContext, partitionPlan);
	}
	else {
		ral::distribution::sendSamplesToMaster(queryContext, std::move(selfSamples), rowSize);

		partitionPlan = ral::distribution::getPartitionPlan(queryContext);
	}

	// Wait for aggregationThread
	std::vector<gdf_column_cpp> aggregatedTable = aggregationTask.get();

	std::vector<int> groupColumnIndices(group_column_indices.size());
  std::iota(groupColumnIndices.begin(), groupColumnIndices.end(), 0);

	std::vector<ral::distribution::NodeColumns> partitions = ral::distribution::partitionData(queryContext, aggregatedTable, groupColumnIndices, partitionPlan);

	ral::distribution::distributePartitions(queryContext, partitions);

	std::vector<ral::distribution::NodeColumns> partitionsToMerge = ral::distribution::collectPartitions(queryContext);
	auto it = std::find_if(partitions.begin(), partitions.end(), [&](ral::distribution::NodeColumns& el) {
			return el.getNode() == CommunicationData::getInstance().getSelfNode();
		});
	// Could "it" iterator be partitions.end()?
	partitionsToMerge.push_back(std::move(*it));

	ral::distribution::aggregationsMerger(partitionsToMerge, groupColumnIndices, aggregation_types, input);
}

void distributed_aggregations_without_groupby(const Context& queryContext, blazing_frame& input, std::vector<int>& group_column_indices, std::vector<gdf_agg_op>& aggregation_types, std::vector<std::string>& aggregation_input_expressions, std::vector<std::string>& aggregation_column_assigned_aliases) {
	using ral::communication::CommunicationData;

	std::vector<gdf_column_cpp> aggregatedTable = compute_aggregations(input,
																																		group_column_indices,
																																		aggregation_types,
																																		aggregation_input_expressions,
																																		aggregation_column_assigned_aliases);

	if (queryContext.isMasterNode(CommunicationData::getInstance().getSelfNode())) {
		std::vector<ral::distribution::NodeColumns> partitionsToMerge = ral::distribution::collectPartitions(queryContext);
		partitionsToMerge.emplace_back(CommunicationData::getInstance().getSelfNode(), std::move(aggregatedTable));

		std::vector<int> groupColumnIndices(group_column_indices.size());
		std::iota(groupColumnIndices.begin(), groupColumnIndices.end(), 0);
		ral::distribution::aggregationsMerger(partitionsToMerge, groupColumnIndices, aggregation_types, input);
	}else{
		std::vector<ral::distribution::NodeColumns> selfPartition;
		selfPartition.emplace_back(queryContext.getMasterNode(), std::move(aggregatedTable));
		ral::distribution::distributePartitions(queryContext, selfPartition);

		input.clear(); // here we are clearing the input, because since there are no group bys, there will only be one result, which will be with the master node
	}
}

void process_aggregate(blazing_frame& input, std::string query_part, const Context* queryContext){
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

	// Get groups
	auto rangeStart = query_part.find("(");
	auto rangeEnd = query_part.rfind(")") - rangeStart - 1;
	std::string combined_expression = query_part.substr(rangeStart + 1, rangeEnd - 1);

	std::vector<int> group_column_indices = get_group_columns(combined_expression);

	// Get aggregations
	std::vector<gdf_agg_op> aggregation_types;
	std::vector<std::string> aggregation_input_expressions;
	std::vector<std::string> aggregation_column_assigned_aliases;
	std::vector<std::string> expressions = get_expressions_from_expression_list(combined_expression);
	for(std::string expr : expressions){
		std::string expression = std::regex_replace(expr, std::regex("^ +| +$|( ) +"), "$1");
		if (expression.find("group=") == std::string::npos)
		{
			gdf_agg_op operation = get_aggregation_operation(expression);
			aggregation_types.push_back(operation);
			aggregation_input_expressions.push_back(get_string_between_outer_parentheses(expression));

			// if the aggregation has an alias, lets capture it here, otherwise we'll figure out what to call the aggregation based on its input
			if (expression.find("EXPR$") == 0)
				aggregation_column_assigned_aliases.push_back("");
			else
				aggregation_column_assigned_aliases.push_back(expression.substr(0, expression.find("=[")));
		}
	}

	if (aggregation_types.size() == 0) {
		if (!queryContext || queryContext->getTotalNodes() <= 1) {
			single_node_groupby_without_aggregations(input, group_column_indices);
		} else{
			distributed_groupby_without_aggregations(*queryContext, input, group_column_indices);
		}
	} else{
		if (!queryContext || queryContext->getTotalNodes() <= 1) {
				single_node_aggregations(input, group_column_indices, aggregation_types, aggregation_input_expressions, aggregation_column_assigned_aliases);
		} else {
				if (group_column_indices.size() == 0) {
					distributed_aggregations_without_groupby(*queryContext, input, group_column_indices, aggregation_types, aggregation_input_expressions, aggregation_column_assigned_aliases);
				} else {
					distributed_aggregations_with_groupby(*queryContext, input, group_column_indices, aggregation_types, aggregation_input_expressions, aggregation_column_assigned_aliases);
				}
		}
	}
}

}  // namespace operators
}  // namespace ral
