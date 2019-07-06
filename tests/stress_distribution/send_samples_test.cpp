#include <tuple>
#include <algorithm>
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

//gtest library
using ::testing::_;
using ::testing::ByRef;
 
#include <communication/network/Server.h>
#include <communication/network/Client.h>
 

namespace {
auto generateHostData(std::size_t size, gdf_dtype dtype) {
  std::mt19937 rng;
  auto Generator = [&rng]() {
    return (rng() % 26) + 65;
  };

  std::size_t data_size = ral::traits::get_data_size_in_bytes(size, dtype);
  std::vector<std::uint8_t> data(data_size);

  std::size_t valid_size = ral::traits::get_bitmask_size_in_bytes(size);
  std::vector<std::uint8_t> valid(valid_size, 0);

  std::generate_n(data.data(), data_size, Generator);
  std::generate_n(valid.data(), valid_size, Generator);

  return std::make_tuple(data, valid);
}

gdf_column_cpp createRalColumn(std::size_t size, gdf_dtype dtype) {
  std::vector<std::uint8_t> data, valid;
  std::tie(data, valid) = generateHostData(size, dtype);

  gdf_column_cpp column;
  auto width = ral::traits::get_dtype_size_in_bytes(dtype);
  column.create_gdf_column(dtype, size, data.data(), /* valid.data(), */ width);

  return column;
}

auto extractDataAndValidsToHost(gdf_column* column) {
  std::size_t data_size = ral::traits::get_data_size_in_bytes(column);
  std::vector<std::uint8_t> data(data_size);
  cudaMemcpy(data.data(), column->data, data_size, cudaMemcpyDeviceToHost);

  std::size_t valid_size = ral::traits::get_bitmask_size_in_bytes(column);
  std::vector<std::uint8_t> valid(valid_size);
  cudaMemcpy(valid.data(), column->valid, valid_size, cudaMemcpyDeviceToHost);

  return std::make_tuple(data, valid);
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
  ral::distribution::NodeSamples node_samples(concreteMessage->getTotalRowSize(),
                                              concreteMessage->getSenderNode(),
                                              std::move(concreteMessage->getSamples()));
  // verify received data
  const std::uint64_t total_row_size = 17;
  std::vector<std::uint8_t> expectedData, expectedValid;
  std::vector<std::uint8_t> data, valid;
  std::vector<gdf_column_cpp> node_columns = node_samples.getColumnsRef();

  // std::tie(expectedData, expectedValid) = generateHostData(total_row_size, GDF_INT16);
  // std::tie(data, valid) = extractDataAndValidsToHost(node_columns[0].get_gdf_column());
  // EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), expectedData.cbegin(), expectedData.cend()));
  // EXPECT_TRUE(std::equal(valid.cbegin(), valid.cend(), expectedValid.cbegin(), expectedValid.cend()));

  // std::tie(expectedData, expectedValid) = generateHostData(total_row_size, GDF_INT64);
  // std::tie(data, valid) = extractDataAndValidsToHost(node_columns[1].get_gdf_column());
  // EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), expectedData.cbegin(), expectedData.cend()));
  // EXPECT_TRUE(std::equal(valid.cbegin(), valid.cend(), expectedValid.cbegin(), expectedValid.cend()));
}

static void ExecWorker() {
  cuInit(0);

  // start server
  Server::start(8001);

  // create context token
  auto context_token = ContextToken::Make(contextTokenValueId);

  // register the context token in the server
  Server::getInstance().registerContext(*context_token);

  // create gdf_column data
  const std::uint64_t total_row_size = 17;
  std::vector<gdf_column_cpp> test_columns;
  test_columns.emplace_back(createRalColumn(total_row_size, GDF_INT8));
  // test_columns.emplace_back(createRalColumn(total_row_size, GDF_INT64));
  {
    // Create message
    Node sender_node(Address::Make("127.0.0.1", 8001, 1234));
    auto message = MessageFactory::createSampleToNodeMaster(*context_token, sender_node, total_row_size, test_columns);

    // Server address
    blazingdb::communication::Node server_node(Address::Make("127.0.0.1", 8000, 1234));

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
