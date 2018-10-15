/*
 ============================================================================
 Name        : testing-libgdf.cu
 Author      : felipe
 Version     :
 Copyright   : Your copyright notice
 Description : MVP
 ============================================================================
 */

#include <cuda_runtime.h>
#include "CalciteInterpreter.h"
#include "ResultSetRepository.h"
#include "DataFrame.h"
#include "Utils.cuh"
#include "Types.h"

#include "ipc/calcite_client.h"
#include "gdf/gdf.h"

#include <tuple>
#include <blazingdb/protocol/api.h>
#include <blazingdb/protocol/interpreter/messages.h>
#include "ral-message.cuh"

using namespace blazingdb::protocol;

using result_pair = std::pair<Status, std::shared_ptr<flatbuffers::DetachedBuffer>>;
using FunctionType = result_pair (*)(uint64_t, Buffer&& buffer);

static result_pair closeConnectionService(uint64_t accessToken, Buffer&& requestPayloadBuffer) {
  std::cout << "accessToken: " << accessToken << std::endl;

  try {
	result_set_repository::get_instance().remove_all_connection_tokens(accessToken);
  } catch (std::runtime_error &error) {
     std::cout << error.what() << std::endl;
     ResponseErrorMessage errorMessage{ std::string{error.what()} };
     return std::make_pair(Status_Error, errorMessage.getBufferData());
  }

  ZeroMessage response{};
  return std::make_pair(Status_Success, response.getBufferData());
}

/*std::shared_ptr<flatbuffers::DetachedBuffer> toBlazingGetResultResponse(blazing_frame result)
{
  flatbuffers::FlatBufferBuilder builder;
  auto metadata =
      interpreter::CreateBlazingMetadata(builder, builder.CreateString("OK"),
                            builder.CreateString("Nothing"), 0.9, 2);
  std::vector<std::string> names{"iron", "man"};
  auto vectorOfNames = builder.CreateVectorOfStrings(names);
  
  std::vector<flatbuffers::Offset<gdf::gdf_column_handler>> values;
  for(auto column : result.get_columns()[0]) {
    gdf::Creategdf_column_handler(builder, column.data(), column.valid(), column.size(), (blazingdb::protocol::gdf::gdf_dtype)column.dtype());
  }

  auto vectorOfValues = builder.CreateVector(values);
  builder.Finish(CreateGetResultResponse(builder, metadata, vectorOfNames,
                                        vectorOfValues));
  std::shared_ptr<flatbuffers::DetachedBuffer> payload =
      std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
}*/

static result_pair getResultService(uint64_t accessToken, Buffer&& requestPayloadBuffer) {
  std::cout << "accessToken: " << accessToken << std::endl;

  interpreter::GetResultRequestMessage requestPayload(requestPayloadBuffer.data());
  std::cout << "resultToken: " << requestPayload.getResultToken() << std::endl;

  // remove from repository using accessToken and resultToken
  blazing_frame result = result_set_repository::get_instance().get_result(accessToken, requestPayload.getResultToken());
  //return std::make_pair(Status_Success, toBlazingGetResultResponse(result));
}

std::tuple<std::vector<std::vector<gdf_column_cpp>>, 
           std::vector<std::string>,
           std::vector<std::vector<std::string>>> toBlazingDataframe(TableGroupDTO request)
{
  std::vector<std::vector<gdf_column_cpp>> input_tables;
  std::vector<std::string> table_names;
  std::vector<std::vector<std::string>> column_names;

  for(auto table : request.tables) {
    table_names.push_back(table.name);
    column_names.push_back(table.columnNames);

    std::vector<gdf_column_cpp> input_table;
    for(auto column : table.columns) {
      input_table.push_back(gdf_column_cpp((void*)column.data, (gdf_valid_type*)column.valid, (::gdf_dtype)column.dtype, (size_t)column.size, (gdf_size_type)column.null_count));
    }
    input_tables.push_back(input_table);
  }

  return std::make_tuple(input_tables, table_names, column_names);
}

static result_pair executePlanService(uint64_t accessToken, Buffer&& requestPayloadBuffer) {
  ::RalRequestMessage requestPayload(requestPayloadBuffer.data());

  // ExecutePlan
  std::cout << "accessToken: " << accessToken << std::endl;
  std::cout << "query: " << requestPayload.getLogicalPlan() << std::endl;
  std::cout << "tableGroup: " << requestPayload.getTableGroup().name << std::endl;
  std::cout << "tables: " << requestPayload.getTableGroup().tables.size() << std::endl;
  std::cout << "tableSize: " << requestPayload.getTableGroup().tables[0].columns[0].size << std::endl;

  std::tuple<std::vector<std::vector<gdf_column_cpp>>, std::vector<std::string>, std::vector<std::vector<std::string>>> request = toBlazingDataframe(requestPayload.getTableGroup());

  uint64_t resultToken = evaluate_query(std::get<0>(request), std::get<1>(request), std::get<2>(request),
                                        requestPayload.getLogicalPlan(), accessToken);
  
  //uint64_t resultToken = 543210L;
  interpreter::NodeConnectionInformationDTO nodeInfo {
      .path = "/tmp/ral.socket",
      .type = interpreter::NodeConnectionType {interpreter::NodeConnectionType_IPC}
  };
  interpreter::ExecutePlanResponseMessage responsePayload{resultToken, nodeInfo};
  return std::make_pair(Status_Success, responsePayload.getBufferData());
}

int main(void)
{ 
  blazingdb::protocol::UnixSocketConnection connection({"/tmp/ral.socket", std::allocator<char>()});
  blazingdb::protocol::Server server(connection);

  std::map<int8_t, FunctionType> services;
  services.insert(std::make_pair(interpreter::MessageType_ExecutePlan, &executePlanService));
  services.insert(std::make_pair(interpreter::MessageType_CloseConnection, &closeConnectionService));
  services.insert(std::make_pair(interpreter::MessageType_GetResult, &getResultService));

  auto interpreterServices = [&services](const blazingdb::protocol::Buffer &requestPayloadBuffer) -> blazingdb::protocol::Buffer {
    RequestMessage request{requestPayloadBuffer.data()};
    std::cout << "header: " << (int)request.messageType() << std::endl;

    auto result = services[request.messageType()] ( request.accessToken(),  request.getPayloadBuffer() );
    ResponseMessage responseObject{result.first, result.second};
    auto bufferedData = responseObject.getBufferData();
    Buffer buffer{bufferedData->data(),
                  bufferedData->size()};
    return buffer;
  };
  server.handle(interpreterServices);

	return 0;
}
