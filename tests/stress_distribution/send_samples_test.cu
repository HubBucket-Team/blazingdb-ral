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

#include <nvstrings/NVCategory.h>
#include <utilities/bit_mask.h>


namespace {
std::mt19937_64 gen;
const int seed = 121;

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
  //column.create_gdf_column(dtype, size, data.data(), width);

  return column;
}

std::string const default_chars = 
	"abcdefghijklmnaoqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
}

std::string random_string(size_t len = 15, std::string const &allowed_chars = default_chars) {
  std::uniform_int_distribution<size_t> dist { 0, allowed_chars.length()-1 };

  std::string ret;
  std::generate_n(std::back_inserter(ret), len, [&] { return allowed_chars[dist(gen)]; });
  return ret;
}

const char ** generate_string_data(gdf_size_type num_rows, size_t length, bool print=false){
  const char ** string_host_data = new const char *[num_rows];

  for(gdf_size_type row_index = 0; row_index < num_rows; row_index++){
    string_host_data[row_index] = new char[length+1];

    std::string rand_string = random_string(length);
    rand_string.push_back(0);
    if(print)
      std::cout<<rand_string<<"\t";
    std::memcpy((void *) string_host_data[row_index],rand_string.c_str(),rand_string.size());
  }
  if(print)
    std::cout<<std::endl;

  return string_host_data;
}

gdf_column * create_nv_category_column_strings(const char ** string_host_data, gdf_size_type num_rows){
  NVCategory* category = NVCategory::create_from_array(string_host_data, num_rows);

  gdf_column * column = new gdf_column;
  int * data;
  EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(gdf_nvstring_category) , 0), RMM_SUCCESS);

  category->get_values( (int *)data, true );
  bit_mask::bit_mask_t * valid;
  bit_mask::create_bit_mask(&valid, num_rows,1);

  gdf_error err = gdf_column_view(column,
      (void *) data,
      (gdf_valid_type *)valid,
      num_rows,
      GDF_STRING_CATEGORY);
  column->dtype_info.category = category;
  column->col_name = nullptr;
  return column;
}

gdf_column_cpp createStringColumn(std::size_t num_values){
  size_t length = 1;
  bool print = true;

  const char ** string_data = generate_string_data(num_values, length, print);

  gdf_column * string_column = create_nv_category_column_strings(string_data, num_values);

  gdf_column_cpp column;
  column.create_gdf_column(string_column);

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
  //test_columns.emplace_back(createRalColumn(total_row_size, GDF_INT64));
  //test_columns.emplace_back(createRalColumn(total_row_size, GDF_INT32));
  //test_columns.emplace_back(createStringColumn(total_row_size));
  {
    // Create message
    Node sender_node(Address::Make("127.0.0.1", 8001, 123));
    auto message = MessageFactory::createSampleToNodeMaster(*context_token, sender_node, total_row_size, test_columns);

    // Server address
    blazingdb::communication::Node server_node(Address::Make("127.0.0.1", 8000, 123));

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