#include "distribution/primitives.h"
#include "cuDF/generator/sample_generator.h"
#include "communication/network/Server.h"
#include "communication/network/Client.h"
#include "communication/messages/ComponentMessages.h"
#include "communication/factory/MessageFactory.h"
#include "communication/CommunicationData.h"
#include "distribution/Exception.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>

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
prepareSamplesForGeneratePivots(
  std::vector<std::vector<gdf_column_cpp>> &samples,
  const std::vector<gdf_size_type> &        tableSizes) {
    assert(samples.size() == tableSizes.size());

    std::vector<double> representativities;
    representativities.reserve(samples.size());

    for (std::size_t i = 0; i < samples.size(); i++) {
        representativities.push_back(double(samples[i][0].size())
                                     / double(tableSizes[i]));
    }

    const gdf_size_type minimumRepresentativity =
      *std::min_element(representativities.cbegin(), representativities.cend());

    for (std::size_t i = 0; i < samples.size(); i++) {
        const double representativenessRatio =
          double(minimumRepresentativity) / representativities[i];

        if (representativenessRatio > thresholdForSubsampling) {
            samples[i] = generateSample(samples[i], representativenessRatio);
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
      throw std::runtime_error("[ERROR] " + std::string{__FUNCTION__} + " -- message type mismatch");
    }

    auto concreteMessage = std::static_pointer_cast<SampleToNodeMasterMessage>(message);
    nodeSamples.emplace_back(concreteMessage->getTotalRowSize(),
                             concreteMessage->getSenderNode(),
                             std::move(concreteMessage->getSamples()));
  }

  return nodeSamples;
}

std::vector<gdf_column_cpp> generatePartitionPlans(std::vector<NodeSamples>& samples){

  // TODO:

  return std::vector<gdf_column_cpp>{};
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
    throw std::runtime_error("[ERROR] " + std::string{__FUNCTION__} + " -- message type mismatch");
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

void sortedMerger(std::vector<NodeColumns>& columns, blazing_frame& output) {
  // TODO: use cudf sorter_merger
}

}  // namespace distribution
}  // namespace ral
