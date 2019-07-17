#include "distribution/primitives.h"
#include "distribution/primitives_util.cuh"
#include "cuDF/generator/sample_generator.h"
#include "communication/network/Server.h"
#include "communication/network/Client.h"
#include "communication/messages/ComponentMessages.h"
#include "communication/factory/MessageFactory.h"
#include "communication/CommunicationData.h"
#include "distribution/Exception.h"
#include "Traits/RuntimeTraits.h"
#include "utilities/RalColumn.h"
#include "utilities/TableWrapper.h"
#include "CalciteExpressionParsing.h"
#include "ColumnManipulation.cuh"
#include <types.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <iostream>
#include "groupby.hpp"
#include "table.hpp"
#include "reduction.hpp"
#include "operators/GroupBy.h"
#include "copying.hpp"
#include "sorted_merge.hpp"
#include "string/nvcategory_util.hpp"

namespace ral {
namespace distribution {
namespace sampling {

double
calculateSampleRatio(gdf_size_type tableSize) {
    return std::ceil(1.0 - std::pow(tableSize/1.0E11, 8E-4));
}

std::vector<gdf_column_cpp>
generateSample(std::vector<gdf_column_cpp> &table, double ratio) {
    std::size_t quantity = std::ceil(table[0].size() * ratio);
    return generateSample(table, quantity);
}

std::vector<std::vector<gdf_column_cpp>>
generateSamples(std::vector<std::vector<gdf_column_cpp>> &tables,
                const std::vector<double> &               ratios) {
    std::vector<std::size_t> quantities;
    quantities.reserve(tables.size());

    for (std::size_t i = 0; i < tables.size(); i++) {
        quantities.push_back(std::ceil(tables[i][0].size() * ratios[i]));
    }

    return generateSamples(tables, quantities);
}

std::vector<gdf_column_cpp>
generateSample(std::vector<gdf_column_cpp> &table, std::size_t quantity) {
    std::vector<gdf_column_cpp> sample;

    gdf_error gdf_status =
      cudf::generator::generate_sample(table, sample, quantity);
    if (GDF_SUCCESS != gdf_status) {
        throw std::runtime_error("[ERROR] " + std::string{__FUNCTION__}
                                 + " -- CUDF: "
                                 + gdf_error_get_name(gdf_status));
    }

    return sample;
}

std::vector<std::vector<gdf_column_cpp>>
generateSamples(std::vector<std::vector<gdf_column_cpp>> &input_tables,
                std::vector<std::size_t> &                quantities) {
    // verify
    if (input_tables.size() != quantities.size()) {
        throw std::runtime_error("[ERROR] " + std::string{__FUNCTION__}
                                 + " -- size mismatch.");
    }

    // output data
    std::vector<std::vector<gdf_column_cpp>> result;

    // make sample for each table
    for (std::size_t k = 0; k < input_tables.size(); ++k) {
        result.emplace_back(generateSample(input_tables[k], quantities[k]));
    }

    // done
    return result;
}

void
normalizeSamples(std::vector<NodeSamples>& samples) {
    std::vector<double> representativities(samples.size());

    for (std::size_t i = 0; i < samples.size(); i++) {
        representativities[i] = (double)samples[i].getColumns()[0].size()
                                / samples[i].getTotalRowSize();
    }

    const double minimumRepresentativity =
      *std::min_element(representativities.cbegin(), representativities.cend());

    for (std::size_t i = 0; i < samples.size(); i++) {
        double representativenessRatio = minimumRepresentativity / representativities[i];

        if (representativenessRatio > THRESHOLD_FOR_SUBSAMPLING) {
            samples[i].setColumns(generateSample(samples[i].getColumnsRef(), representativenessRatio));
        }
    }
}

}  // namespace sampling
}  // namespace distribution
}  // namespace ral


namespace ral {
namespace distribution {

void sendSamplesToMaster(const Context& context, std::vector<gdf_column_cpp>&& samples, std::size_t total_row_size) {
    // Get master node
    const Node& master_node = context.getMasterNode();

    // Get self node
    using CommunicationData = ral::communication::CommunicationData;
    const Node& self_node = CommunicationData::getInstance().getSelfNode();

    // Get context token
    const auto& context_token = context.getContextToken();

    // Create message
    using MessageFactory = ral::communication::messages::Factory;
    using SampleToNodeMasterMessage = ral::communication::messages::SampleToNodeMasterMessage;
    auto message = MessageFactory::createSampleToNodeMaster(context_token, self_node, total_row_size, std::move(samples));

    // Send message to master
    using Client = ral::communication::network::Client;
    Client::send(master_node, message);
}

std::vector<NodeSamples> collectSamples(const Context& context) {
  using ral::communication::network::Server;
  using ral::communication::messages::SampleToNodeMasterMessage;

  std::vector<NodeSamples> nodeSamples;
  auto& contextToken = context.getContextToken();
  auto size = context.getWorkerNodes().size();
  for (int k = 0; k < size; ++k) {
    auto message = Server::getInstance().getMessage(contextToken, SampleToNodeMasterMessage::getMessageID());

    if (message->getMessageTokenValue() != SampleToNodeMasterMessage::getMessageID()) {
      throw createMessageMismatchException(__FUNCTION__,
                                           SampleToNodeMasterMessage::getMessageID(),
                                           message->getMessageTokenValue());
    }

    auto concreteMessage = std::static_pointer_cast<SampleToNodeMasterMessage>(message);
    nodeSamples.emplace_back(concreteMessage->getTotalRowSize(),
                             concreteMessage->getSenderNode(),
                             std::move(concreteMessage->getSamples()));
  }

  return nodeSamples;
}

std::vector<gdf_column_cpp> generatePartitionPlans(const Context& context, std::vector<NodeSamples>& samples, std::vector<int8_t>& sortOrderTypes){
  // Concat
  size_t totalConcatsOperations = samples[0].getColumnsRef().size();
  int outputRowSize = 0;
  std::vector<std::vector<gdf_column*>> columnsToConcatArray(totalConcatsOperations);
  for(size_t i = 0; i < samples.size(); i++)
  {
    auto& columns = samples[i].getColumnsRef();
    outputRowSize += columns[0].size();

    assert(columns.size() == totalConcatsOperations);
    for(size_t j = 0; j < totalConcatsOperations; j++)
    {
      columnsToConcatArray[j].push_back(columns[j].get_gdf_column());
    }
  }

  auto& tempCols = samples[0].getColumnsRef();
  std::vector<gdf_column_cpp> concatSamples(totalConcatsOperations);
  for(size_t i = 0; i < concatSamples.size(); i++) {
    auto& col = tempCols[i];
    if (std::any_of(columnsToConcatArray[i].begin(), columnsToConcatArray[i].end(), [](auto* c){ return c->valid != nullptr; })) {
      concatSamples[i].create_gdf_column(col.dtype(), outputRowSize, nullptr, get_width_dtype(col.dtype()), col.name());
    } else {
      concatSamples[i].create_gdf_column(col.dtype(), outputRowSize, nullptr, nullptr, get_width_dtype(col.dtype()), col.name());
    }

    CUDF_CALL( gdf_column_concat(concatSamples[i].get_gdf_column(),
                                columnsToConcatArray[i].data(),
                                columnsToConcatArray[i].size()) );
  }

  // std::cout << "After Concat\n";
  // for(auto& p : concatSamples)
  // {
  //     print_gdf_column(p.get_gdf_column());
  // }

  // Sort
  std::vector<gdf_column*> rawConcatSamples(concatSamples.size());
  for(size_t i = 0; i < concatSamples.size(); i++)
  {
    rawConcatSamples[i] = concatSamples[i].get_gdf_column();
  }

  gdf_column_cpp ascDescCol;
	ascDescCol.create_gdf_column(GDF_INT8, sortOrderTypes.size(), sortOrderTypes.data(), get_width_dtype(GDF_INT8), "");

  gdf_column_cpp sortedIndexCol;
	sortedIndexCol.create_gdf_column(GDF_INT32, outputRowSize, nullptr, get_width_dtype(GDF_INT32), "");

	gdf_context gdfContext;
	gdfContext.flag_null_sort_behavior = GDF_NULL_AS_LARGEST; // Nulls are are treated as largest

	CUDF_CALL( gdf_order_by(rawConcatSamples.data(),
                          (int8_t*)(ascDescCol.get_gdf_column()->data),
                          rawConcatSamples.size(),
                          sortedIndexCol.get_gdf_column(),
                          &gdfContext) );

  std::vector<gdf_column_cpp> sortedSamples(concatSamples.size());
 	for(size_t i = 0; i < sortedSamples.size(); i++) {
    auto& col = concatSamples[i];
    if (col.valid()) {
      sortedSamples[i].create_gdf_column(col.dtype(), col.size(), nullptr, get_width_dtype(col.dtype()), col.name());
    } else {
      sortedSamples[i].create_gdf_column(col.dtype(), col.size(), nullptr, nullptr, get_width_dtype(col.dtype()), col.name());
    }

    materialize_column(
      concatSamples[i].get_gdf_column(),
      sortedSamples[i].get_gdf_column(),
      sortedIndexCol.get_gdf_column()
    );
    sortedSamples[i].update_null_count();
	}

  // std::cout << "After Sort\n";
  // for(auto& p : sortedSamples)
  // {
  //     print_gdf_column(p.get_gdf_column());
  // }

  // Gather
  std::vector<gdf_column*> rawSortedSamples(sortedSamples.size());
  std::vector<gdf_column*> rawPivots(sortedSamples.size());
  std::vector<gdf_column_cpp> pivots(sortedSamples.size());
 	for(size_t i = 0; i < sortedSamples.size(); i++) {
    auto& col = sortedSamples[i];
    if (col.valid()) {
      pivots[i].create_gdf_column(col.dtype(), context.getTotalNodes() - 1, nullptr, get_width_dtype(col.dtype()), col.name());
    } else {
      pivots[i].create_gdf_column(col.dtype(), context.getTotalNodes() - 1, nullptr, nullptr, get_width_dtype(col.dtype()), col.name());
    }
    
    rawPivots[i] = pivots[i].get_gdf_column();
    rawSortedSamples[i] = sortedSamples[i].get_gdf_column();
	}

  cudf::table srcTable{rawSortedSamples.data(), (gdf_size_type)rawSortedSamples.size()};
  cudf::table destTable{rawPivots.data(), (gdf_size_type)rawPivots.size()};

  int step = outputRowSize / context.getTotalNodes();
  gdf_column_cpp gatherMap;
	gatherMap.create_gdf_column(GDF_INT32, context.getTotalNodes() - 1, nullptr, get_width_dtype(GDF_INT32), "");
	gdf_sequence(static_cast<int32_t*>(gatherMap.get_gdf_column()->data), gatherMap.size(), step, step);

  std::cout << "Gather Map\n";
  print_gdf_column(gatherMap.get_gdf_column());

  cudf::gather(&srcTable, (gdf_index_type*)(gatherMap.get_gdf_column()->data), &destTable);
  
  for(int i = 0; i < destTable.num_columns(); ++i){
    auto* srcCol = srcTable.get_column(i);
    auto* dstCol = destTable.get_column(i);
    if(dstCol->dtype == GDF_STRING_CATEGORY){
      nvcategory_gather(dstCol,static_cast<NVCategory *>(srcCol->dtype_info.category));
    }
  }

  std::cout << "After Gather\n";
  for(auto& p : pivots)
  {
      print_gdf_column(p.get_gdf_column());
  }

  return pivots;
}

void distributePartitionPlan(const Context& context, std::vector<gdf_column_cpp>& pivots){
  using ral::communication::network::Client;
  using ral::communication::CommunicationData;
  using ral::communication::messages::Factory;

  auto message = Factory::createPartitionPivotsMessage(context.getContextToken(),
                                                      CommunicationData::getInstance().getSelfNode(),
                                                      pivots);
  auto workers = context.getWorkerNodes();
  for(auto& workerNode : workers)
  {
    Client::send(*workerNode, message);
  }
}

std::vector<gdf_column_cpp> getPartitionPlan(const Context& context){
  using ral::communication::network::Server;
  using ral::communication::messages::PartitionPivotsMessage;

  auto message = Server::getInstance().getMessage(context.getContextToken(), PartitionPivotsMessage::getMessageID());

  if (message->getMessageTokenValue() != PartitionPivotsMessage::getMessageID()) {
    throw createMessageMismatchException(__FUNCTION__,
                                         PartitionPivotsMessage::getMessageID(),
                                         message->getMessageTokenValue());
  }

  auto concreteMessage = std::static_pointer_cast<PartitionPivotsMessage>(message);

  return std::move(concreteMessage->getColumns());
}

std::vector<NodeColumns> split_data_into_NodeColumns(const Context& context, const std::vector<gdf_column_cpp>& table, const gdf_column_cpp & indexes){

  std::vector<std::vector<gdf_column*>> split_table(table.size()); // this will be [colInd][splitInd]
    for (std::size_t k = 0; k < table.size(); ++k) {
      split_table[k] = cudf::split(*(table[k].get_gdf_column()), static_cast<gdf_index_type*>(indexes.data()), indexes.size());
    }

    // get nodes
    auto nodes = context.getAllNodes();

    // generate NodeColumns
    std::vector<NodeColumns> array_node_columns;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        std::vector<gdf_column_cpp> columns(table.size());
        for (std::size_t k = 0; k < table.size(); ++k) {
          split_table[k][i]->col_name = nullptr;
            columns[k].create_gdf_column(split_table[k][i]);
            columns[k].set_name(table[k].name());
        }

        array_node_columns.emplace_back(*nodes[i], std::move(columns));
    }
    return array_node_columns;
}


std::vector<NodeColumns> partitionData(const Context& context,
                                       std::vector<gdf_column_cpp>& table,
                                       std::vector<int>& searchColIndices,
                                       std::vector<gdf_column_cpp>& pivots) {
    // verify input
    if (pivots.size() == 0) {
        throw std::runtime_error("The pivots array is empty");
    }

    if (pivots.size() != searchColIndices.size()) {
        throw std::runtime_error("The pivots and searchColIndices vectors don't have the same size");
    }

    auto& pivot = pivots[0];
    {
        std::size_t size = pivot.size();

        // verify the size of the pivots.
        for (std::size_t k = 1; k < pivots.size(); ++k) {
            if (size != pivots[k].size()) {
                throw std::runtime_error("The pivots don't have the same size");
            }
        }

        // verify the size in pivots and nodes
        auto nodes = context.getAllNodes();
        if (nodes.size() != (size + 1)) {
            throw std::runtime_error("The size of the nodes needs to be the same as the size of the pivots plus one");
        }
    }

    gdf_size_type table_column_size = table[0].size();
    if (table_column_size == 0)
    {
      std::vector<NodeColumns> array_node_columns;
      auto nodes = context.getAllNodes();
      for (std::size_t i = 0; i < nodes.size(); ++i) {
          std::vector<gdf_column_cpp> columns = table;
          array_node_columns.emplace_back(*nodes[i], std::move(columns));
      }
      return array_node_columns;
    }

    // create output column
    gdf_column_cpp indexes = ral::utilities::create_zero_column(pivot.size(), GDF_SIZE_TYPE);

    // apply gdf_multisearch
    ral::utilities::TableWrapper haystack(table, searchColIndices);
    ral::utilities::TableWrapper needles(pivots);

    CUDF_CALL( gdf_multisearch(indexes.get_gdf_column(),
                              haystack.getColumns(),
                              needles.getColumns(),
                              haystack.getQuantity(),
                              true,   // find_first_greater
                              false,  // nulls_appear_before_values
                              true) );  // use_haystack_length_for_not_found

    // std::cout << "multisearch indices\n";
    // print_gdf_column(indexes.get_gdf_column());

    sort_indices(indexes);
    
    return split_data_into_NodeColumns(context, table, indexes);    
}

void distributePartitions(const Context& context, std::vector<NodeColumns>& partitions){
  using ral::communication::network::Client;
  using ral::communication::messages::Factory;
  using ral::communication::CommunicationData;

  const auto& self_node = CommunicationData::getInstance().getSelfNode();
  for(auto& nodeColumn : partitions)
  {
    if (nodeColumn.getNode() == self_node) {
      continue;
    }
    auto message = Factory::createColumnDataMessage(context.getContextToken(),
                                                    self_node,
                                                    std::move(nodeColumn.getColumns()));
    Client::send(nodeColumn.getNode(), message);
  }
}

std::vector<NodeColumns> collectPartitions(const Context& context) {
    using ral::communication::network::Server;
    using ral::communication::messages::ColumnDataMessage;

    // Get the numbers of rals in the query
    auto number_rals = context.getAllNodes().size() - 1;

    // Create return value
    std::vector<NodeColumns> node_columns;

    // Get message from the server
    const auto& context_token = context.getContextToken();
    while (0 < number_rals) {
        auto message = Server::getInstance().getMessage(context_token, ColumnDataMessage::getMessageID());
        number_rals--;

        if (message->getMessageTokenValue() != ColumnDataMessage::getMessageID()) {
            throw createMessageMismatchException(__FUNCTION__,
                                                 ColumnDataMessage::getMessageID(),
                                                 message->getMessageTokenValue());
        }

        auto column_message = std::static_pointer_cast<ColumnDataMessage>(message);
        node_columns.emplace_back(message->getSenderNode(),
                                  std::move(column_message->getColumns()));
    }
    return node_columns;
}

void sortedMerger(std::vector<NodeColumns>& columns, std::vector<int8_t>& sortOrderTypes, std::vector<int>& sortColIndices, blazing_frame& output) {
  rmm::device_vector<int8_t> ascDesc(sortOrderTypes);

  std::vector<gdf_column_cpp> leftCols = columns[0].getColumns();
  std::vector<gdf_column*> rawLeftCols(leftCols.size());
  std::transform(leftCols.begin(), leftCols.end(), rawLeftCols.begin(), [&](gdf_column_cpp& el) {
    return el.get_gdf_column();
  });

  cudf::table leftTable(rawLeftCols.data(), rawLeftCols.size());
  for(size_t i = 1; i < columns.size(); i++)
  {
    if (columns[i].getColumnsRef()[0].size() == 0) {
      continue;
    }

    std::vector<gdf_column_cpp> rightCols = columns[i].getColumns();
    std::vector<gdf_column*> rawRightCols(rightCols.size());
    std::transform(rightCols.begin(), rightCols.end(), rawRightCols.begin(), [&](gdf_column_cpp& el) {
      return el.get_gdf_column();
    });

    leftTable = cudf::sorted_merge(leftTable,
                                  cudf::table(rawRightCols.data(), rawRightCols.size()),
                                  sortColIndices,
                                  ascDesc);

    // std::for_each(leftTable.begin(), leftTable.end(), [](auto* c){
    //   print_gdf_column(c);
    // });
  }

  for (size_t i = 0; i < leftTable.num_columns(); i++) {
    gdf_column * col = leftTable.get_column(i);
    col->col_name = nullptr; // TODO: Delete this after updating cudf to properly
    std::string colName = leftCols[i].name(); // zero initialize gdf_columns
    leftCols[i].create_gdf_column(col);
    leftCols[i].set_name(colName);
  }

  output.clear();
  output.add_table(leftCols);
}

std::vector<gdf_column_cpp> generatePartitionPlansGroupBy(const Context& context, std::vector<NodeSamples>& samples){
  // Concat
  size_t totalConcatsOperations = samples[0].getColumnsRef().size();
  int outputRowSize = 0;
  std::vector<std::vector<gdf_column*>> columnsToConcatArray{totalConcatsOperations};
  for(size_t i = 0; i < samples.size(); i++)
  {
    auto& columns = samples[i].getColumnsRef();
    outputRowSize += columns[0].size();

    assert(columns.size() == totalConcatsOperations);
    for(size_t j = 0; j < totalConcatsOperations; j++)
    {
      columnsToConcatArray[j].push_back(columns[j].get_gdf_column());
    }
  }

  auto& tempCols = samples[0].getColumnsRef();
  std::vector<gdf_column_cpp> concatSamples{totalConcatsOperations};
  for(size_t i = 0; i < concatSamples.size(); i++)
  {
    auto& col = tempCols[i];
    if (std::any_of(columnsToConcatArray[i].begin(), columnsToConcatArray[i].end(), [](auto* c){ return c->valid != nullptr; })) {
      concatSamples[i].create_gdf_column(col.dtype(), outputRowSize, nullptr, get_width_dtype(col.dtype()), col.name());
    } else {
      concatSamples[i].create_gdf_column(col.dtype(), outputRowSize, nullptr, nullptr, get_width_dtype(col.dtype()), col.name());
    }

    CUDF_CALL( gdf_column_concat(concatSamples[i].get_gdf_column(),
                                columnsToConcatArray[i].data(),
                                columnsToConcatArray[i].size()) );
  }

  // std::cout << "After Concat\n";
  // for(auto& p : concatSamples)
  // {
  //     print_gdf_column(p.get_gdf_column());
  // }

  // // Get uniques
  // std::vector<gdf_column*> rawConcatSamples(concatSamples.size());
  // std::vector<gdf_column_cpp> tempGroupedSamples(concatSamples.size());
  // std::vector<gdf_column*> rawTempGroupedSamples(concatSamples.size());
  // for(size_t i = 0; i < concatSamples.size(); i++)
  // {
  //   rawConcatSamples[i] = concatSamples[i].get_gdf_column();
  //   tempGroupedSamples[i].create_gdf_column(concatSamples[i].dtype(),
	// 																			concatSamples[i].size(),
	// 																			nullptr,
	// 																			get_width_dtype(concatSamples[i].dtype()),
	// 																			concatSamples[i].name());
  //   rawTempGroupedSamples[i] = tempGroupedSamples[i].get_gdf_column();
  // }

	// gdf_column_cpp groupedIndexCol;
  // groupedIndexCol.create_gdf_column(GDF_INT32, outputRowSize, nullptr, get_width_dtype(GDF_INT32), "");

  // gdf_size_type indexColNumRows = 0;

  // gdf_context ctxt;
  // ctxt.flag_null_sort_behavior = GDF_NULL_AS_LARGEST; //  Nulls are are treated as largest
  // ctxt.flag_groupby_include_nulls = 1; // Nulls are treated as values in group by keys where NULL == NULL (SQL style)

  // std::vector<int> groupColumnIndices(concatSamples.size());
  // std::iota(groupColumnIndices.begin(), groupColumnIndices.end(), 0);

  // CUDF_CALL( gdf_group_by_without_aggregations(rawConcatSamples.size(),
  //                                             rawConcatSamples.data(),
  //                                             groupColumnIndices.size(),
  //                                             groupColumnIndices.data(),
  //                                             rawTempGroupedSamples.data(),
  //                                             (gdf_size_type*)(groupedIndexCol.get_gdf_column()->data),
  //                                             &indexColNumRows,
  //                                             &ctxt));
  // groupedIndexCol.resize(indexColNumRows);

  // std::vector<gdf_column_cpp> groupedSamples(concatSamples.size());
  // for(size_t i = 0; i < groupedSamples.size(); i++){
  //   groupedSamples[i].create_gdf_column(concatSamples[i].dtype(),
	// 																			groupedIndexCol.size(),
	// 																			nullptr,
	// 																			get_width_dtype(concatSamples[i].dtype()),
	// 																			concatSamples[i].name());
  //   materialize_column(rawTempGroupedSamples[i],
  //                     groupedSamples[i].get_gdf_column(),
  //                     groupedIndexCol.get_gdf_column());
  //   groupedSamples[i].update_null_count();
  // }

/*****/

    std::vector<int> groupColumnIndices(concatSamples.size());
    std::iota(groupColumnIndices.begin(), groupColumnIndices.end(), 0);
    std::vector<gdf_column_cpp> groupedSamples = ral::operators::groupby_without_aggregations(concatSamples, groupColumnIndices); 
    size_t number_of_groups = groupedSamples[0].size();

/****/

  // std::cout << "After Group\n";
  // for(auto& p : groupedSamples)
  // {
  //     print_gdf_column(p.get_gdf_column());
  // }

  // Sort
  std::vector<gdf_column*> rawGroupedSamples{groupedSamples.size()};
  for(size_t i = 0; i < groupedSamples.size(); i++)
  {
    rawGroupedSamples[i] = groupedSamples[i].get_gdf_column();
  }

  gdf_column_cpp sortedIndexCol;
  sortedIndexCol.create_gdf_column(GDF_INT32, number_of_groups, nullptr, get_width_dtype(GDF_INT32), "");
	
	gdf_context gdfContext;
	gdfContext.flag_null_sort_behavior = GDF_NULL_AS_LARGEST; // Nulls are are treated as largest

	CUDF_CALL( gdf_order_by(rawGroupedSamples.data(),
                          nullptr,
                          rawGroupedSamples.size(),
                          sortedIndexCol.get_gdf_column(),
                          &gdfContext) );

  std::vector<gdf_column_cpp> sortedSamples(groupedSamples.size());
 	for(size_t i = 0; i < sortedSamples.size(); i++) {
    auto& col = groupedSamples[i];
    if (col.valid()) {
      sortedSamples[i].create_gdf_column(col.dtype(), col.size(), nullptr, get_width_dtype(col.dtype()), col.name());
    } else {
      sortedSamples[i].create_gdf_column(col.dtype(), col.size(), nullptr, nullptr, get_width_dtype(col.dtype()), col.name());
    }

    materialize_column(
      groupedSamples[i].get_gdf_column(),
      sortedSamples[i].get_gdf_column(),
      sortedIndexCol.get_gdf_column()
    );
    sortedSamples[i].update_null_count();
	}

  // std::cout << "After Sort\n";
  // for(auto& p : sortedSamples)
  // {
  //     print_gdf_column(p.get_gdf_column());
  // }

  // Gather
  std::vector<gdf_column*> rawSortedSamples{sortedSamples.size()};
  std::vector<gdf_column*> rawPivots{sortedSamples.size()};
  std::vector<gdf_column_cpp> pivots{sortedSamples.size()};
 	for(size_t i = 0; i < sortedSamples.size(); i++) {
    auto& col = sortedSamples[i];
    if (col.valid()) {
      pivots[i].create_gdf_column(col.dtype(), context.getTotalNodes() - 1, nullptr, get_width_dtype(col.dtype()), col.name());
    } else {
      pivots[i].create_gdf_column(col.dtype(), context.getTotalNodes() - 1, nullptr, nullptr, get_width_dtype(col.dtype()), col.name());
    }

    rawPivots[i] = pivots[i].get_gdf_column();
    rawSortedSamples[i] = sortedSamples[i].get_gdf_column();
	}

  cudf::table srcTable{rawSortedSamples.data(), (gdf_size_type)rawSortedSamples.size()};
  cudf::table destTable{rawPivots.data(), (gdf_size_type)rawPivots.size()};

  int step = sortedIndexCol.size() / context.getTotalNodes();
  gdf_column_cpp gatherMap;
	gatherMap.create_gdf_column(GDF_INT32, context.getTotalNodes() - 1, nullptr, get_width_dtype(GDF_INT32), "");
	gdf_sequence(static_cast<int32_t*>(gatherMap.get_gdf_column()->data), gatherMap.size(), step, step);

  std::cout << "Gather Map\n";
  print_gdf_column(gatherMap.get_gdf_column());

  cudf::gather(&srcTable, (gdf_index_type*)(gatherMap.get_gdf_column()->data), &destTable);

  for(int i = 0; i < destTable.num_columns(); ++i){
    auto* srcCol = srcTable.get_column(i);
    auto* dstCol = destTable.get_column(i);
    if(dstCol->dtype == GDF_STRING_CATEGORY){
      nvcategory_gather(dstCol,static_cast<NVCategory *>(srcCol->dtype_info.category));
    }
  }

  std::cout << "After Gather\n";
  for(auto& p : pivots)
  {
      print_gdf_column(p.get_gdf_column());
  }

  return pivots;
}

void groupByWithoutAggregationsMerger(std::vector<NodeColumns>& groups, const std::vector<int>& groupColIndices, blazing_frame& output){
  // Concat
  size_t totalConcatsOperations = groupColIndices.size();
  int outputRowSize = 0;
  std::vector<std::vector<gdf_column*>> columnsToConcatArray(totalConcatsOperations);
  for(size_t i = 0; i < groups.size(); i++)
  {
    auto& columns = groups[i].getColumnsRef();
    if (columns[0].size() == 0) {
      continue;
    }

    outputRowSize += columns[0].size();

    assert(columns.size() == totalConcatsOperations);
    for(size_t j = 0; j < totalConcatsOperations; j++)
    {
      columnsToConcatArray[j].push_back(columns[j].get_gdf_column());
    }
  }

  if (outputRowSize == 0)
  {
    output.clear();
    std::vector<gdf_column_cpp> output_table = groups[0].getColumnsRef();
    output.add_table(std::move(output_table));
    return;
  }

  std::vector<gdf_column_cpp> concatGroups(totalConcatsOperations);
  for(size_t i = 0; i < concatGroups.size(); i++)
  {
    auto* tempGdfCol = columnsToConcatArray[i][0];
    concatGroups[i].create_gdf_column(tempGdfCol->dtype, outputRowSize, nullptr, get_width_dtype(tempGdfCol->dtype), std::string(tempGdfCol->col_name));
    CUDF_CALL( gdf_column_concat(concatGroups[i].get_gdf_column(),
                                columnsToConcatArray[i].data(),
                                columnsToConcatArray[i].size()) );
  }

  // std::cout << "After Concat\n";
  // for(auto& p : concatGroups)
  // {
  //   print_gdf_column(p.get_gdf_column());
  // }

  // Do groupBy
  // size_t nCols = concatGroups.size();

  // std::vector<gdf_column*> rawCols(nCols);
  // std::vector<gdf_column_cpp> outputColumns(nCols);
  // std::vector<gdf_column*> rawOutputColumns(nCols);
  // for(size_t i = 0; i < nCols; i++){
  //   rawCols[i] = concatGroups[i].get_gdf_column();
  //   outputColumns[i].create_gdf_column(concatGroups[i].dtype(),
	// 																		outputRowSize,
	// 																		nullptr,
	// 																		get_width_dtype(concatGroups[i].dtype()),
	// 																		concatGroups[i].name());
  //   rawOutputColumns[i] = outputColumns[i].get_gdf_column();
  // }

	// gdf_column_cpp indexCol;
  // indexCol.create_gdf_column(GDF_INT32, outputRowSize, nullptr, get_width_dtype(GDF_INT32), "");

  // gdf_size_type indexColNumRows = 0;

  // gdf_context ctxt;
  // ctxt.flag_null_sort_behavior = GDF_NULL_AS_LARGEST; //  Nulls are are treated as largest
  // ctxt.flag_groupby_include_nulls = 1; // Nulls are treated as values in group by keys where NULL == NULL (SQL style)

  // CUDF_CALL( gdf_group_by_without_aggregations(rawCols.size(),
  //                                             rawCols.data(),
  //                                             groupColIndices.size(),
  //                                             groupColIndices.data(),
  //                                             rawOutputColumns.data(),
  //                                             (gdf_size_type*)(indexCol.get_gdf_column()->data),
  //                                             &indexColNumRows,
  //                                             &ctxt));
  // indexCol.resize(indexColNumRows);

  // std::vector<gdf_column_cpp> groupedOutput(nCols);
  // for(size_t i = 0; i < nCols; i++){
  //   groupedOutput[i].create_gdf_column(concatGroups[i].dtype(),
	// 																			indexCol.size(),
	// 																			nullptr,
	// 																			get_width_dtype(concatGroups[i].dtype()),
	// 																			concatGroups[i].name());
  //   materialize_column(rawOutputColumns[i],
  //                     groupedOutput[i].get_gdf_column(),
  //                     indexCol.get_gdf_column());
  //   groupedOutput[i].update_null_count();
  // }

  /**************/
  std::vector<gdf_column_cpp> groupedOutput = ral::operators::groupby_without_aggregations(concatGroups, groupColIndices);    

  /**************/



  // std::cout << "After Merge\n";
  // for(auto& p : groupedOutput)
  // {
  //   print_gdf_column(p.get_gdf_column());
  // }

	output.clear();
	output.add_table(groupedOutput);
}

namespace {
  // TODO DUPLICATE IN GROUP BY
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

// TODO DUPLICATE IN GROUP BY
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

// TODO DUPLICATE IN GROUP BY
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
}

void aggregationsMerger(std::vector<NodeColumns>& aggregations, const std::vector<int>& groupColIndices, const std::vector<gdf_agg_op>& aggregationTypes, blazing_frame& output){
  // std::vector<NodeColumns>& aggregations => [groupedCols, aggregatedCols]

  // Concat
  size_t totalConcatsOperations = groupColIndices.size() + aggregationTypes.size();
  int outputRowSize = 0;
  std::vector<std::vector<gdf_column*>> columnsToConcatArray(totalConcatsOperations);
  for(size_t i = 0; i < aggregations.size(); i++)
  {
    auto& columns = aggregations[i].getColumnsRef();
    if (columns[0].size() == 0) {
      continue;
    }
    outputRowSize += columns[0].size();

    assert(columns.size() == totalConcatsOperations);
    for(size_t j = 0; j < totalConcatsOperations; j++)
    {
      columnsToConcatArray[j].push_back(columns[j].get_gdf_column());
    }
  }

  if (outputRowSize == 0)
  {
    output.clear();
    std::vector<gdf_column_cpp> output_table = aggregations[0].getColumnsRef();
    output.add_table(std::move(output_table));
    return;
  }
  
  std::vector<gdf_column_cpp> concatAggregations(totalConcatsOperations);
  for(size_t i = 0; i < concatAggregations.size(); i++)
  {
    auto* tempGdfCol = columnsToConcatArray[i][0];
    concatAggregations[i].create_gdf_column(tempGdfCol->dtype, outputRowSize, nullptr, get_width_dtype(tempGdfCol->dtype), std::string(tempGdfCol->col_name));
    CUDF_CALL( gdf_column_concat(concatAggregations[i].get_gdf_column(),
                                columnsToConcatArray[i].data(),
                                columnsToConcatArray[i].size()) );
  }

  // std::cout << "After Concat\n";
  // for(auto& p : concatAggregations)
  // {
  //   print_gdf_column(p.get_gdf_column());
  // }

  // Do aggregations
	std::vector<gdf_column*> rawGroupByColumns(groupColIndices.size());
	std::vector<gdf_column_cpp> groupedColumns(groupColIndices.size());
	std::vector<gdf_column*> rawGroupedColumns(groupColIndices.size());
	for(size_t i = 0; i < groupColIndices.size(); i++){
		gdf_column_cpp& inputColumn = concatAggregations[groupColIndices[i]];

		rawGroupByColumns[i] = inputColumn.get_gdf_column();

		groupedColumns[i].create_gdf_column(inputColumn.dtype(), outputRowSize, nullptr, get_width_dtype(inputColumn.dtype()), inputColumn.name());
		rawGroupedColumns[i] = groupedColumns[i].get_gdf_column();
	}

  // when we are merging COUNT aggregations, we want to SUM them, not use COUNT
  std::vector<gdf_agg_op> modAggregationTypes(aggregationTypes.size());
  for(size_t i = 0; i < aggregationTypes.size(); i++){
    modAggregationTypes[i] = aggregationTypes[i] == GDF_COUNT ? GDF_SUM : aggregationTypes[i];
  }

	// If we have no groups you will output only one row
	size_t aggregation_size = (groupColIndices.size() == 0 ? 1 : outputRowSize);

	std::vector<gdf_column_cpp> aggregatedColumns(modAggregationTypes.size());
	for(size_t i = 0; i < modAggregationTypes.size(); i++){
    // Use "groupColIndices.size() + i" because concatAggregations has the same layout as std::vector<NodeColumns>& aggregations
		gdf_column_cpp& aggregationInput = concatAggregations[groupColIndices.size() + i];

		if (groupColIndices.size() == 0) {
			aggregations_without_groupby(modAggregationTypes[i], aggregationInput, aggregatedColumns[i], aggregationInput.dtype(), aggregationInput.name());
		}else{
      aggregatedColumns[i].create_gdf_column(aggregationInput.dtype(), aggregation_size, nullptr, get_width_dtype(aggregationInput.dtype()), aggregationInput.name());
			aggregations_with_groupby(modAggregationTypes[i],
																rawGroupByColumns,
																aggregationInput,
																rawGroupedColumns,
																aggregatedColumns[i]);
		}

		//so that subsequent iterations won't be too large
		aggregation_size = aggregatedColumns[i].size();
	}

  std::vector<gdf_column_cpp> outputTable(std::move(groupedColumns));
	outputTable.insert(
		outputTable.end(),
		std::make_move_iterator(aggregatedColumns.begin()),
		std::make_move_iterator(aggregatedColumns.end())
	);

  // std::cout << "After Merge\n";
  // for(auto& p : outputTable)
  // {
  //   print_gdf_column(p.get_gdf_column());
  // }

	output.clear();
	output.add_table(outputTable);
}

}  // namespace distribution
}  // namespace ral


namespace ral {
namespace distribution {

std::vector<gdf_column_cpp> generateOutputColumns(gdf_size_type column_quantity,
                                                  gdf_size_type column_size,
                                                  std::vector<gdf_column_cpp>& table) {
    // Create outcome
    std::vector<gdf_column_cpp> result(column_quantity);

    // Create columns
    for (gdf_size_type k = 0; k < column_quantity; ++k) {
        result[k] = ral::utilities::create_column(column_size, table[k].dtype());
    }

    // Done
    return result;
}

std::vector<NodeColumns> generateJoinPartitions(const Context& context,
                                                std::vector<gdf_column_cpp>& table,
                                                std::vector<int>& columnIndices) {
    assert(table.size() != 0);

    std::vector<gdf_column*> raw_input_table_col_ptrs(table.size());
    std::transform(table.begin(), table.end(), raw_input_table_col_ptrs.begin(), [](auto& cpp_col){
      return cpp_col.get_gdf_column();
    });
    cudf::table input_table_wrapper(raw_input_table_col_ptrs);
    
    if (input_table_wrapper.num_rows() == 0) {
        std::vector<NodeColumns> result;
        auto nodes = context.getAllNodes();
        for (gdf_size_type k = 0; k < nodes.size(); ++k) {
            std::vector<gdf_column_cpp> columns = table;
            result.emplace_back(*nodes[k], std::move(columns));
        }
        return result;
    }

    // Generate partition offset vector
    gdf_size_type number_nodes = context.getTotalNodes();
    std::vector<gdf_index_type> partition_offset(number_nodes);

    // Preallocate output columns
    std::vector<gdf_column_cpp> output_columns = generateOutputColumns(input_table_wrapper.num_columns(),
                                                                       input_table_wrapper.num_rows(),
                                                                       table);
    std::vector<gdf_column*> raw_output_table_col_ptrs(table.size());
    std::transform(output_columns.begin(), output_columns.end(), raw_output_table_col_ptrs.begin(), [](auto& cpp_col){
      return cpp_col.get_gdf_column();
    });
    cudf::table output_table_wrapper(raw_output_table_col_ptrs);

    // Execute operation
    CUDF_CALL( gdf_hash_partition(input_table_wrapper.num_columns(),
                                  input_table_wrapper.begin(),
                                  columnIndices.data(),
                                  columnIndices.size(),
                                  number_nodes,
                                  output_table_wrapper.begin(),
                                  partition_offset.data(),
                                  gdf_hash_func::GDF_HASH_MURMUR3) );

    for(int i = 0; i < output_table_wrapper.num_columns(); ++i){
      auto* srcCol = input_table_wrapper.get_column(i);
      auto* dstCol = output_table_wrapper.get_column(i);
      if(dstCol->dtype == GDF_STRING_CATEGORY){
        nvcategory_gather(dstCol,static_cast<NVCategory *>(srcCol->dtype_info.category));
      }
    }      

    // Erase input table
    table.clear();

    // lets get the split indices. These are all the partition_offset, except for the first since its just 0
    gdf_column_cpp indexes;
    indexes.create_gdf_column(GDF_INT32, number_nodes - 1, partition_offset.data() + 1, nullptr, get_width_dtype(GDF_INT32), "");
    
    return split_data_into_NodeColumns(context, output_columns, indexes);
}

} // namespace distribution
} // namespace ral
