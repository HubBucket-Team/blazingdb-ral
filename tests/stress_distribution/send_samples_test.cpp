#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "CalciteExpressionParsing.h"
#include "blazingdb/communication/network/Status.h"
#include "distribution/Exception.h"
#include "distribution/NodeColumns.h"
#include "distribution/NodeSamples.h"
#include "distribution/primitives.h"
#include "communication/CommunicationData.h"
#include "communication/factory/MessageFactory.h"
#include "Traits/RuntimeTraits.h"
#include "utilities/RalColumn.h"
#include "cuio/parquet/util/bit_util.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include "GDFColumn.cuh"
#include <distribution/NodeSamples.h>

//gtest library
using ::testing::_;
using ::testing::ByRef;
 
#include <communication/network/Server.h>
#include <communication/network/Client.h>
 

namespace {
gdf_column_cpp createRalColumn(std::size_t size, gdf_dtype dtype) {
  std::mt19937 rng;
  auto Generator = [&rng]() {
    return (rng() % 26) + 65;
  };

  std::size_t data_size = ral::traits::get_data_size_in_bytes(size, dtype);
  std::vector<std::uint8_t> data;
  data.resize(data_size);

  std::size_t valid_size = ral::traits::get_bitmask_size_in_bytes(size);
  std::vector<std::uint8_t> valid;
  valid.resize(valid_size);

  std::generate_n(data.data(), data_size, Generator);
  std::generate_n(valid.data(), valid_size, Generator);

  gdf_column_cpp column;
  auto width = ral::traits::get_dtype_size_in_bytes(dtype);
  column.create_gdf_column(dtype, size, data.data(), valid.data(), width);

  return column;
}

std::vector<std::uint8_t> get_data(gdf_column* column) {
  std::vector<std::uint8_t> result;

  std::size_t data_size = ral::traits::get_data_size_in_bytes(column);
  result.resize(data_size);
  cudaMemcpy(result.data(), column->data, data_size, cudaMemcpyDeviceToHost);

  return result;
}

std::vector<std::uint8_t> get_valid(gdf_column* column) {
  std::vector<std::uint8_t> result;

  std::size_t valid_size = ral::traits::get_bitmask_size_in_bytes(column);
  result.resize(valid_size);
  cudaMemcpy(result.data(), column->valid, valid_size, cudaMemcpyDeviceToHost);

  return result;
}
}

static constexpr blazingdb::communication::network::Server::ContextTokenValue contextTokenValueId = 1230;

using Server = ral::communication::network::Server;
using ContextToken = blazingdb::communication::ContextToken;
using blazingdb::communication::Address;
using MessageFactory = ral::communication::messages::Factory;
namespace Messages = ral::communication::messages;
using Node = blazingdb::communication::Node;

static void ExecMaster() {
  cuInit(0);

  // start server
  Server::start(8000);

  // create context token
  auto context_token = ContextToken::Make(contextTokenValueId);

  // register the context token in the server
  Server::getInstance().registerContext(*context_token);

  // Receive message from the client
  auto message = Server::getInstance().getMessage(*context_token, Messages::SampleToNodeMasterMessage::getMessageID());

  auto concreteMessage = std::static_pointer_cast<Messages::SampleToNodeMasterMessage>(message);
  ral::distribution::NodeSamples node_samples{concreteMessage->getTotalRowSize(),
                           concreteMessage->getSenderNode(),
                           std::move(concreteMessage->getSamples())};
  // verify received data
}

static void ExecWorker() {
  cuInit(0);

  // start server
  Server::start(8001);

  // create context token
  auto context_token = ContextToken::Make(contextTokenValueId);

  // register the context token in the server
  Server::getInstance().registerContext(*context_token);

  const std::uint64_t total_row_size = 16;


  // create gdf_column data
  std::vector<gdf_column_cpp> test_columns;
  test_columns.emplace_back(createRalColumn(total_row_size, GDF_INT16));
  test_columns.emplace_back(createRalColumn(total_row_size, GDF_INT64));
  {
    // Create message
    Node sender_node(Address::Make("127.0.0.1", 8001));
    auto message = MessageFactory::createSampleToNodeMaster(*context_token, sender_node, total_row_size, test_columns);

    // Server address
    blazingdb::communication::Node server_node(Address::Make("127.0.0.1", 8000));

    // Send message to the server
    using ral::communication::network::Client;
    Client::send(server_node, message);
  }
}


TEST(SendSamplesTest, TwoProcesses) {
  pid_t pid = fork();
  if (pid) {
    ExecMaster();
  } else {
    ExecWorker();
    std::exit(0);
  }
}


TEST(SendSamplesTest, Master) {
  ExecMaster();
}

TEST(SendSamplesTest, Worker) {
  ExecWorker();
}