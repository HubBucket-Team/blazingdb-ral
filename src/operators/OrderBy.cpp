#include <iostream>
#include <thread>
#include <functional>
#include <blazingdb/io/Library/Logging/Logger.h>
#include <blazingdb/io/Util/StringUtil.h>
#include "config/GPUManager.cuh"
#include "OrderBy.h"
#include "CodeTimer.h"
#include "CalciteExpressionParsing.h"
#include "distribution/primitives.h"
#include "communication/CommunicationData.h"
#include "ColumnManipulation.cuh"
#include "GDFColumn.cuh"

namespace ral {
namespace operators {

namespace {
using blazingdb::communication::Context;
} // namespace

const std::string LOGICAL_SORT_TEXT = "LogicalSort";
const std::string ASCENDING_ORDER_SORT_TEXT = "ASC";
const std::string DESCENDING_ORDER_SORT_TEXT = "DESC";

bool is_sort(std::string query_part){
	return (query_part.find(LOGICAL_SORT_TEXT) != std::string::npos);
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

void sort(blazing_frame& input, std::vector<gdf_column*>& rawCols, std::vector<int8_t>& sortOrderTypes, std::vector<gdf_column_cpp>& sortedTable){
	static CodeTimer timer;
	timer.reset();

	gdf_column_cpp asc_desc_col;
	asc_desc_col.create_gdf_column(GDF_INT8, sortOrderTypes.size(), sortOrderTypes.data(), get_width_dtype(GDF_INT8), "");

	gdf_column_cpp index_col;
	index_col.create_gdf_column(GDF_INT32, input.get_num_rows_in_table(0), nullptr, get_width_dtype(GDF_INT32), "");

	gdf_context context;
	context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST; // Nulls are are treated as largest

	CUDF_CALL( gdf_order_by(rawCols.data(),
			(int8_t*)(asc_desc_col.get_gdf_column()->data),
			rawCols.size(),
			index_col.get_gdf_column(),
			&context));

	Library::Logging::Logger().logInfo("-> Sort sub block 2 took " + std::to_string(timer.getDuration()) + " ms");

	timer.reset();

	for(int i = 0; i < sortedTable.size(); i++){
		materialize_column(
			input.get_column(i).get_gdf_column(),
			sortedTable[i].get_gdf_column(),
			index_col.get_gdf_column()
		);
		sortedTable[i].update_null_count();
	}

	Library::Logging::Logger().logInfo("-> Sort sub block 3 took " + std::to_string(timer.getDuration()) + " ms");
}

void single_node_sort(blazing_frame& input, std::vector<gdf_column*>& rawCols, std::vector<int8_t>& sortOrderTypes) {
	std::vector<gdf_column_cpp> sortedTable(input.get_size_column(0));
	for(int i = 0; i < sortedTable.size();i++){
		auto& input_col = input.get_column(i);
		if (input_col.valid())
			sortedTable[i].create_gdf_column(input_col.dtype(), input_col.size(), nullptr, get_width_dtype(input_col.dtype()), input_col.name());
		else 
			sortedTable[i].create_gdf_column(input_col.dtype(), input_col.size(), nullptr, nullptr, get_width_dtype(input_col.dtype()), input_col.name());
	}

	sort(input, rawCols, sortOrderTypes, sortedTable);

	input.clear();
	input.add_table(sortedTable);
}

void distributed_sort(const Context& queryContext, blazing_frame& input, std::vector<gdf_column_cpp>& cols, std::vector<gdf_column*>& rawCols, std::vector<int8_t>& sortOrderTypes, std::vector<int>& sortColIndices) {
	using ral::communication::CommunicationData;

	std::vector<gdf_column_cpp> sortedTable(input.get_size_column(0));
	for(int i = 0; i < sortedTable.size();i++){
		auto& input_col = input.get_column(i);
		if (input_col.valid())
			sortedTable[i].create_gdf_column(input_col.dtype(), input_col.size(), nullptr, get_width_dtype(input_col.dtype()), input_col.name());
		else 
			sortedTable[i].create_gdf_column(input_col.dtype(), input_col.size(), nullptr, nullptr, get_width_dtype(input_col.dtype()), input_col.name());
	}

	size_t rowSize = input.get_num_rows_in_table(0);

	std::vector<gdf_column_cpp> selfSamples = ral::distribution::sampling::generateSample(cols, 0.1);

	std::thread sortThread{[](blazing_frame& input, std::vector<gdf_column*>& rawCols, std::vector<int8_t>& sortOrderTypes, std::vector<gdf_column_cpp>& sortedTable){
		ral::config::GPUManager::getInstance().setDevice();
		sort(input, rawCols, sortOrderTypes, sortedTable);
	}, std::ref(input), std::ref(rawCols), std::ref(sortOrderTypes), std::ref(sortedTable)};
	// sort(input, rawCols, sortOrderTypes, sortedTable);

	std::vector<gdf_column_cpp> partitionPlan;
	if (queryContext.isMasterNode(CommunicationData::getInstance().getSelfNode())) {
		std::vector<ral::distribution::NodeSamples> samples = ral::distribution::collectSamples(queryContext);
    samples.emplace_back(rowSize, CommunicationData::getInstance().getSelfNode(), std::move(selfSamples));

		partitionPlan = ral::distribution::generatePartitionPlans(queryContext, samples, sortOrderTypes);

    ral::distribution::distributePartitionPlan(queryContext, partitionPlan);
	}
	else {
		ral::distribution::sendSamplesToMaster(queryContext, std::move(selfSamples), rowSize);

		partitionPlan = ral::distribution::getPartitionPlan(queryContext);
	}

	// Wait for sortThread
	sortThread.join();

	std::vector<ral::distribution::NodeColumns> partitions = ral::distribution::partitionData(queryContext, sortedTable, sortColIndices, partitionPlan, true, sortOrderTypes);

	ral::distribution::distributePartitions(queryContext, partitions);

	std::vector<ral::distribution::NodeColumns> partitionsToMerge = ral::distribution::collectPartitions(queryContext);
	auto it = std::find_if(partitions.begin(), partitions.end(), [&](ral::distribution::NodeColumns& el) {
			return el.getNode() == CommunicationData::getInstance().getSelfNode();
		});
	// Could "it" iterator be partitions.end()?
	partitionsToMerge.push_back(std::move(*it));

	ral::distribution::sortedMerger(partitionsToMerge, sortOrderTypes, sortColIndices, input);
}

void process_sort(blazing_frame & input, std::string query_part, const Context* queryContext){
	static CodeTimer timer;
	timer.reset();
	std::cout<<"about to process sort"<<std::endl;

	auto fetchLimit = query_part.find("fetch");
	if(fetchLimit != std::string::npos) {
		throw std::runtime_error{"In evaluate_split_query function: unsupported limit clause"};
	}

	//LogicalSort(sort0=[$4], sort1=[$7], dir0=[ASC], dir1=[ASC])
	auto rangeStart = query_part.find("(");
	auto rangeEnd = query_part.rfind(")") - rangeStart - 1;
	std::string combined_expression = query_part.substr(rangeStart + 1, rangeEnd - 1);

	size_t num_sort_columns = count_string_occurrence(combined_expression,"sort");

	std::vector<gdf_column_cpp> cols(num_sort_columns);
	std::vector<gdf_column*> rawCols(num_sort_columns);
	std::vector<int8_t> sortOrderTypes(num_sort_columns);
	std::vector<int> sortColIndices(num_sort_columns);
	for(int i = 0; i < num_sort_columns; i++){
		int sort_column_index = get_index(get_named_expression(combined_expression, "sort" + std::to_string(i)));
		gdf_column_cpp col = input.get_column(sort_column_index).clone();
		rawCols[i] = col.get_gdf_column();
		cols[i] = std::move(col);
		sortOrderTypes[i] = (get_named_expression(combined_expression, "dir" + std::to_string(i)) == DESCENDING_ORDER_SORT_TEXT);
		sortColIndices[i] = sort_column_index;
	}

	Library::Logging::Logger().logInfo("-> Sort sub block 1 took " + std::to_string(timer.getDuration()) + " ms");

	if (!queryContext || queryContext->getTotalNodes() <= 1) {
		single_node_sort(input, rawCols, sortOrderTypes);
	}
	else {
		distributed_sort(*queryContext, input, cols, rawCols, sortOrderTypes, sortColIndices);
	}
}

}  // namespace operators
}  // namespace ral
