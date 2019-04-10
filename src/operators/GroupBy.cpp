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

void perform_avg(gdf_column* column_output, gdf_column* column_input){
	uint64_t avg_sum = 0;
	gdf_dtype dtype;
	size_t dtype_size;

	// Step 1
	dtype = column_input->dtype;
	dtype_size = get_width_dtype(dtype);

	gdf_column_cpp column_avg;
	column_avg.create_gdf_column(dtype, 1, nullptr, dtype_size);

	gdf_column_cpp temp;
	temp.create_gdf_column(dtype, gdf_reduction_get_intermediate_output_size(), nullptr, dtype_size, "");

	CUDF_CALL( gdf_sum(column_input, temp.get_gdf_column()->data, temp.size()) );
	CheckCudaErrors(cudaMemcpy(&avg_sum, temp.get_gdf_column()->data, dtype_size, cudaMemcpyDeviceToHost));

	// Step 2
	uint64_t avg_count = column_input->size - column_input->null_count;

	dtype = column_output->dtype;
	dtype_size = get_width_dtype(dtype);

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
		// TODO: felipe percy noboa see upgrade to uints
		// else if (Ral::Traits::is_dtype_unsigned(dtype)) {
		//     uint64_t result = (uint64_t) avg_sum / (uint64_t) avg_count;
		//     CheckCudaErrors(cudaMemcpy(column_output->data, &result, dtype_size, cudaMemcpyHostToDevice));
		// }
	}
	else {
		throw std::runtime_error{"In perform_avg function: unsupported dtype"};
	}
}

std::vector<gdf_column_cpp> groupby_without_aggregations(blazing_frame& input, std::vector<int>& group_column_indices){
	size_t nCols = input.get_size_column(0);
  size_t nRows = input.get_column(0).size();

  std::vector<gdf_column*> raw_cols(nCols);
  std::vector<gdf_column_cpp> output_columns(nCols);
  std::vector<gdf_column*> raw_output_columns(nCols);
  for(size_t i = 0; i < nCols; i++){
    raw_cols[i] = input.get_column(i).get_gdf_column();
    output_columns[i].create_gdf_column(input.get_column(i).dtype(),
																				nRows,
																				nullptr,
																				get_width_dtype(input.get_column(i).dtype()),
																				input.get_column(i).name());
    raw_output_columns[i] = output_columns[i].get_gdf_column();
  }

	gdf_column_cpp index_col;
  index_col.create_gdf_column(GDF_INT32, nRows, nullptr, get_width_dtype(GDF_INT32), "");

  gdf_size_type index_col_num_rows = 0;

  gdf_context ctxt;
  ctxt.flag_null_sort_behavior = GDF_NULL_AS_LARGEST; //  Nulls are are treated as largest
  ctxt.flag_groupby_include_nulls = 1; // Nulls are treated as values in group by keys where NULL == NULL (SQL style)

  CUDF_CALL( gdf_group_by_without_aggregations(raw_cols.size(),
                                              raw_cols.data(),
                                              group_column_indices.size(),
                                              group_column_indices.data(),
                                              raw_output_columns.data(),
                                              (gdf_size_type*)(index_col.get_gdf_column()->data),
                                              &index_col_num_rows,
                                              &ctxt));
  index_col.resize(index_col_num_rows);

  std::vector<gdf_column_cpp> grouped_output(nCols);
  for(size_t i = 0; i < nCols; i++){
    grouped_output[i].create_gdf_column(input.get_column(i).dtype(),
																				index_col.size(),
																				nullptr,
																				get_width_dtype(input.get_column(i).dtype()),
																				input.get_column(i).name());
    materialize_column(raw_output_columns[i],
                      grouped_output[i].get_gdf_column(),
                      index_col.get_gdf_column());
    grouped_output[i].update_null_count();
  }

	return grouped_output;
}

void single_node_groupby_without_aggregations(blazing_frame& input, std::vector<int>& group_column_indices){
	std::vector<gdf_column_cpp> grouped_table = groupby_without_aggregations(input, group_column_indices);

  input.clear();
  input.add_table(grouped_table);
}

void distributed_groupby_without_aggregations(const Context& queryContext, blazing_frame& input, std::vector<int>& group_column_indices){
	using ral::communication::CommunicationData;

	std::vector<gdf_column_cpp> group_columns(group_column_indices.size());
	for(size_t i = 0; i < group_column_indices.size(); i++){
		group_columns[i] = input.get_column(group_column_indices[i]);
	}

	size_t rowSize = input.get_column(0).size();

	std::vector<gdf_column_cpp> selfSamples = ral::distribution::sampling::generateSample(group_columns, 0.1);

	auto groupByTask = std::async(std::launch::async,
																groupby_without_aggregations,
																std::ref(input),
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

	std::vector<ral::distribution::NodeColumns> partitionsToMerge = ral::distribution::collectPartition(queryContext);
	auto it = std::find_if(partitions.begin(), partitions.end(), [&](ral::distribution::NodeColumns& el) {
			return el.getNode() == CommunicationData::getInstance().getSelfNode();
		});
	// Could "it" iterator be partitions.end()?
	partitionsToMerge.push_back(std::move(*it));

	ral::distribution::groupByMerger(partitionsToMerge, group_column_indices, input);
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

void aggregations_without_groupby(gdf_agg_op agg_op, gdf_column_cpp& aggregation_input, gdf_column_cpp& output_column){
	gdf_column_cpp temp;
	switch(agg_op){
		case GDF_SUM:
			if (aggregation_input.size() == 0) {
				// Set output_column data to invalid
				CheckCudaErrors(cudaMemset(output_column.valid(), (gdf_valid_type)0, output_column.get_valid_size()));
				output_column.update_null_count();
				break;
			}
			temp.create_gdf_column(output_column.dtype(), gdf_reduction_get_intermediate_output_size(), nullptr, get_width_dtype(output_column.dtype()), "");
			CUDF_CALL(gdf_sum(aggregation_input.get_gdf_column(), temp.data(), temp.size()));
			CheckCudaErrors(cudaMemcpy(output_column.data(), temp.data(), 1 * get_width_dtype(output_column.dtype()), cudaMemcpyDeviceToDevice));
			break;
		case GDF_MIN:
			if (aggregation_input.size() == 0) {
				// Set output_column data to invalid
				CheckCudaErrors(cudaMemset(output_column.valid(), (gdf_valid_type)0, output_column.get_valid_size()));
				output_column.update_null_count();
				break;
			}
			temp.create_gdf_column(output_column.dtype(), gdf_reduction_get_intermediate_output_size(), nullptr, get_width_dtype(output_column.dtype()), "");
			CUDF_CALL(gdf_min(aggregation_input.get_gdf_column(), temp.data(), temp.size()));
			CheckCudaErrors(cudaMemcpy(output_column.data(), temp.data(), 1 * get_width_dtype(output_column.dtype()), cudaMemcpyDeviceToDevice));
			break;
		case GDF_MAX:
			if (aggregation_input.size() == 0) {
				// Set output_column data to invalid
				CheckCudaErrors(cudaMemset(output_column.valid(), (gdf_valid_type)0, output_column.get_valid_size()));
				output_column.update_null_count();
				break;
			}
			temp.create_gdf_column(output_column.dtype(), gdf_reduction_get_intermediate_output_size(), nullptr, get_width_dtype(output_column.dtype()), "");
			CUDF_CALL(gdf_max(aggregation_input.get_gdf_column(), temp.data(), temp.size()));
			CheckCudaErrors(cudaMemcpy(output_column.data(), temp.data(), 1 * get_width_dtype(output_column.dtype()), cudaMemcpyDeviceToDevice));
			break;
		case GDF_AVG:
			if (aggregation_input.size() == 0) {
				// Set output_column data to invalid
				CheckCudaErrors(cudaMemset(output_column.valid(), (gdf_valid_type)0, output_column.get_valid_size()));
				output_column.update_null_count();
				break;
			}
			perform_avg(output_column.get_gdf_column(), aggregation_input.get_gdf_column());
			break;
		case GDF_COUNT:
		{
			// output dtype is GDF_UINT64
			// defined in 'get_aggregation_output_type' function.
			uint64_t result = aggregation_input.size() - aggregation_input.null_count();
			CheckCudaErrors(cudaMemcpy(output_column.data(), &result, sizeof(uint64_t), cudaMemcpyHostToDevice));
			break;
		}
		case GDF_COUNT_DISTINCT:
		{
			// output dtype is GDF_UINT64
			// defined in 'get_aggregation_output_type' function.
			uint64_t result = aggregation_input.size() - aggregation_input.null_count();
			CheckCudaErrors(cudaMemcpy(output_column.data(), &result, sizeof(uint64_t), cudaMemcpyHostToDevice));
			break;
		}
	}
}

std::vector<gdf_column_cpp> compute_aggregations(blazing_frame& input, std::vector<int>& group_column_indices, std::vector<gdf_agg_op>& aggregation_types, std::vector<std::string>& aggregation_input_expressions, std::vector<std::string>& aggregation_column_assigned_aliases){
	size_t row_size = input.get_column(0).size();

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

		gdf_dtype output_type = get_aggregation_output_type(aggregation_input.dtype(), aggregation_types[i], group_column_indices.size());

		// if the aggregation was given an alias lets use it, otherwise we'll name it based on the aggregation and input
		std::string output_column_name = (aggregation_column_assigned_aliases[i] == ""
																			? (aggregator_to_string(aggregation_types[i]) + "(" + aggregation_input.name() + ")")
																			: aggregation_column_assigned_aliases[i]);
		output_columns_aggregations[i].create_gdf_column(output_type, aggregation_size, nullptr, get_width_dtype(output_type), output_column_name);

		if (group_column_indices.size() == 0) {
			aggregations_without_groupby(aggregation_types[i], aggregation_input, output_columns_aggregations[i]);
		}else{
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

	size_t rowSize = input.get_column(0).size();

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

	std::vector<ral::distribution::NodeColumns> partitionsToMerge = ral::distribution::collectPartition(queryContext);
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
		std::vector<ral::distribution::NodeColumns> partitionsToMerge = ral::distribution::collectPartition(queryContext);
		partitionsToMerge.emplace_back(CommunicationData::getInstance().getSelfNode(), std::move(aggregatedTable));

		std::vector<int> groupColumnIndices(group_column_indices.size());
		std::iota(groupColumnIndices.begin(), groupColumnIndices.end(), 0);
		ral::distribution::aggregationsMerger(partitionsToMerge, groupColumnIndices, aggregation_types, input);
	}else{
		std::vector<ral::distribution::NodeColumns> selfPartition;
		selfPartition.emplace_back(queryContext.getMasterNode(), std::move(aggregatedTable));
		ral::distribution::distributePartitions(queryContext, selfPartition);

		// TODO: clear input
		// input.clear();
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
    }else{
      distributed_groupby_without_aggregations(*queryContext, input, group_column_indices);
    }
	}else{
		if (!queryContext || queryContext->getTotalNodes() <= 1) {
      single_node_aggregations(input, group_column_indices, aggregation_types, aggregation_input_expressions, aggregation_column_assigned_aliases);
    }else{
			if (group_column_indices.size() == 0) {
				distributed_aggregations_without_groupby(*queryContext, input, group_column_indices, aggregation_types, aggregation_input_expressions, aggregation_column_assigned_aliases);
			}
			else {
	      distributed_aggregations_with_groupby(*queryContext, input, group_column_indices, aggregation_types, aggregation_input_expressions, aggregation_column_assigned_aliases);
			}
    }
	}
}

}  // namespace operators
}  // namespace ral
