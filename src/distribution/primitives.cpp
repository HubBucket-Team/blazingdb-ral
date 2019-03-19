#include "distribution/primitives.h"
#include "cuDF/generator/sample_generator.h"
#include "communication/network/Server.h"
#include "communication/network/Client.h"
#include "communication/messages/ComponentMessages.h"
#include "communication/factory/MessageFactory.h"
#include "communication/CommunicationData.h"
#include "distribution/Exception.h"
#include "CalciteExpressionParsing.h"
#include "ColumnManipulation.cuh"
#include <copying.hpp>
#include <types.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <iostream>

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
    std::vector<double> representativities{samples.size()};

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

std::vector<NodeColumns> collectPartition(const Context& context) {
    // Alias
    using ColumnDataMessage = ral::communication::messages::ColumnDataMessage;

    // Get the numbers of rals in the query
    auto number_rals = context.getAllNodes().size() - 1;

    // Create return value
    std::vector<NodeColumns> node_columns;

    // Get message from the server
    const auto& context_token = context.getContextToken();
    auto& server = ral::communication::network::Server::getInstance();
    while (0 < number_rals) {
        auto message = server.getMessage(context_token);
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

std::vector<NodeSamples> collectSamples(const Context& context) {
  using ral::communication::network::Server;
  using ral::communication::messages::SampleToNodeMasterMessage;

  std::vector<NodeSamples> nodeSamples;
  auto& contextToken = context.getContextToken();
  auto size = context.getWorkerNodes().size();
  for (int k = 0; k < size; ++k) {
    auto message = Server::getInstance().getMessage(contextToken);

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
    concatSamples[i].create_gdf_column(tempCols[i].dtype(), outputRowSize, nullptr, get_width_dtype(tempCols[i].dtype()), tempCols[i].name());
    CUDF_CALL( gdf_column_concat(concatSamples[i].get_gdf_column(),
                                columnsToConcatArray[i].data(),
                                columnsToConcatArray[i].size()) );
  }

  std::cout << "After Concat\n";
  for(auto& p : concatSamples)
  {
      print_gdf_column(p.get_gdf_column());
  }

  // Sort
  std::vector<gdf_column*> rawCols{concatSamples.size()};
  for(size_t i = 0; i < concatSamples.size(); i++)
  {
    rawCols[i] = concatSamples[i].get_gdf_column();
  }

  gdf_column_cpp ascDescCol;
	ascDescCol.create_gdf_column(GDF_INT8, sortOrderTypes.size(), sortOrderTypes.data(), get_width_dtype(GDF_INT8), "");

  gdf_column_cpp indexCol;
	indexCol.create_gdf_column(GDF_INT32, outputRowSize, nullptr, get_width_dtype(GDF_INT32), "");

	gdf_context gdfContext;
	gdfContext.flag_null_sort_behavior = GDF_NULL_AS_LARGEST; // Nulls are are treated as largest

	CUDF_CALL( gdf_order_by(rawCols.data(),
                          (int8_t*)(ascDescCol.get_gdf_column()->data),
                          rawCols.size(),
                          indexCol.get_gdf_column(),
                          &gdfContext) );

  std::vector<gdf_column_cpp> sortedSamples{concatSamples.size()};
 	for(size_t i = 0; i < sortedSamples.size(); i++) {
    sortedSamples[i].create_gdf_column(concatSamples[i].dtype(),
																			concatSamples[i].size(),
																			nullptr,
																			get_width_dtype(concatSamples[i].dtype()),
																			concatSamples[i].name());
    materialize_column(
      concatSamples[i].get_gdf_column(),
      sortedSamples[i].get_gdf_column(),
      indexCol.get_gdf_column()
    );
    sortedSamples[i].update_null_count();
	}

  std::cout << "After Sort\n";
  for(auto& p : sortedSamples)
  {
      print_gdf_column(p.get_gdf_column());
  }

  // Gather
  std::vector<gdf_column*> rawSortedSamples{sortedSamples.size()};
  std::vector<gdf_column*> rawPivots{sortedSamples.size()};
  std::vector<gdf_column_cpp> pivots{sortedSamples.size()};
 	for(size_t i = 0; i < sortedSamples.size(); i++) {
    pivots[i].create_gdf_column(sortedSamples[i].dtype(),
                                context.getTotalNodes() - 1,
                                nullptr,
                                get_width_dtype(sortedSamples[i].dtype()),
                                sortedSamples[i].name());
    rawPivots[i] = pivots[i].get_gdf_column();
    rawSortedSamples[i] = sortedSamples[i].get_gdf_column();
	}

  cudf::table srcTable{rawSortedSamples.data(), rawSortedSamples.size()};
  cudf::table destTable{rawPivots.data(), rawPivots.size()};

  int step = outputRowSize / context.getTotalNodes();
  gdf_column_cpp gatherMap;
	gatherMap.create_gdf_column(GDF_INT32, context.getTotalNodes() - 1, nullptr, get_width_dtype(GDF_INT32), "");
	gdf_sequence(static_cast<int32_t*>(gatherMap.get_gdf_column()->data), gatherMap.size(), step, step);

  std::cout << "Gather Map\n";
  print_gdf_column(gatherMap.get_gdf_column());

  cudf::gather(&srcTable, (gdf_index_type*)(gatherMap.get_gdf_column()->data), &destTable);

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

  auto message = Factory::createColumnDataMessage(context.getContextToken(),
                                                  CommunicationData::getInstance().getSelfNode(),
                                                  std::move(pivots));
  auto workers = context.getWorkerNodes();
  for(auto& workerNode : workers)
  {
    Client::send(*workerNode, message);
  }
}

std::vector<gdf_column_cpp> getPartitionPlan(const Context& context){
  using ral::communication::network::Server;
  using ral::communication::messages::ColumnDataMessage;

  auto message = Server::getInstance().getMessage(context.getContextToken());

  if (message->getMessageTokenValue() != ColumnDataMessage::getMessageID()) {
    throw createMessageMismatchException(__FUNCTION__,
                                         ColumnDataMessage::getMessageID(),
                                         message->getMessageTokenValue());
  }

  auto concreteMessage = std::static_pointer_cast<ColumnDataMessage>(message);

  return std::move(concreteMessage->getColumns());
}

std::vector<NodeColumns> partitionData(const Context& context, std::vector<gdf_column_cpp>& table, std::vector<gdf_column_cpp>& pivots){
  // TODO

  return std::vector<NodeColumns>{};
}

void distributePartitions(const Context& context, std::vector<NodeColumns>& partitions){
  using ral::communication::network::Client;
  using ral::communication::messages::Factory;
  using ral::communication::CommunicationData;

  const auto& self_node = CommunicationData::getInstance().getSelfNode();
  for(auto& nodeColumn : partitions)
  {
    auto message = Factory::createColumnDataMessage(context.getContextToken(),
                                                    self_node,
                                                    std::move(nodeColumn.getColumns()));
    Client::send(nodeColumn.getNode(), message);
  }
}

void sortedMerger(std::vector<NodeColumns>& columns, std::vector<int8_t>& sortOrderTypes, std::vector<int>& sortColIndices, blazing_frame& output) {
  gdf_column_cpp ascDescCol;
	ascDescCol.create_gdf_column(GDF_INT8, sortOrderTypes.size(), sortOrderTypes.data(), get_width_dtype(GDF_INT8), "");

  gdf_column_cpp sortByColIndices;
	sortByColIndices.create_gdf_column(GDF_INT32, sortColIndices.size(), sortColIndices.data(), get_width_dtype(GDF_INT32), "");

  std::vector<gdf_column_cpp> leftCols = columns[0].getColumns();
  std::vector<gdf_column*> rawLeftCols{leftCols.size()};
  std::transform(leftCols.begin(), leftCols.end(), rawLeftCols.begin(), [&](gdf_column_cpp& el) {
    return el.get_gdf_column();
  });

  for(size_t i = 1; i < columns.size(); i++)
  {
    std::vector<gdf_column_cpp> rightCols = columns[i].getColumns();
    std::vector<gdf_column*> rawRightCols{rightCols.size()};
    std::transform(rightCols.begin(), rightCols.end(), rawRightCols.begin(), [&](gdf_column_cpp& el) {
      return el.get_gdf_column();
    });

    // Create output cols
    std::vector<gdf_column_cpp> sortedColumns{leftCols.size()};
    std::vector<gdf_column*> rawSortedColumns{sortedColumns.size()};
    for(size_t j = 0; j < sortedColumns.size(); j++) {
      sortedColumns[j].create_gdf_column(leftCols[j].dtype(),
                                        leftCols[j].size() + rightCols[j].size(),
                                        nullptr,
                                        get_width_dtype(leftCols[j].dtype()),
                                        leftCols[j].name());
      rawSortedColumns[j] = sortedColumns[j].get_gdf_column();
    }


    CUDF_CALL( gdf_sorted_merge(rawLeftCols.data(),
                                rawRightCols.data(),
                                rawLeftCols.size(),
                                sortByColIndices.get_gdf_column(),
                                ascDescCol.get_gdf_column(),
                                rawSortedColumns.data()));

    leftCols = std::move(sortedColumns);
    rawLeftCols = std::move(rawSortedColumns);
  }

  output.clear();
  output.add_table(leftCols);
}

}  // namespace distribution
}  // namespace ral
