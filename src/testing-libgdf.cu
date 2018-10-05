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

#include <blazingdb/protocol/api.h>
#include <blazingdb/protocol/interpreter/messages.h>

using namespace blazingdb::protocol;

using result_pair = std::pair<Status, std::shared_ptr<flatbuffers::DetachedBuffer>>;
using FunctionType = result_pair (*)(uint64_t, const uint8_t* buffer);

static result_pair closeConnectionService(uint64_t accessToken, const uint8_t* requestPayloadBuffer) {
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

static result_pair getResultService(uint64_t accessToken, const uint8_t* requestPayloadBuffer) {
   std::cout << "accessToken: " << accessToken << std::endl;

  interpreter::GetResultRequestMessage requestPayload(requestPayloadBuffer);
  std::cout << "resultToken: " << requestPayload.getResultToken() << std::endl;

  // remove from repository using accessToken and resultToken

  ZeroMessage response{}; // @todo: GetResultResponseMessage
  return std::make_pair(Status_Success, response.getBufferData());
}


static result_pair executePlanService(uint64_t accessToken, const uint8_t* requestPayloadBuffer)   {
  interpreter::DMLRequestMessage requestPayload(requestPayloadBuffer);

  std::cout << "accessToken: " << accessToken << std::endl;
  std::cout << "query: " << requestPayload.getLogicalPlan() << std::endl;

  query_token_t resultToken;

  try {

	std::vector<std::vector<gdf_column_cpp> > input_tables;
	std::vector<std::string> table_names;
	std::vector<std::vector<std::string>> column_names;
	std::string logicalPlan = requestPayload.getLogicalPlan();

	resultToken = evaluate_query(input_tables, table_names, column_names,
									logicalPlan, accessToken);

  } catch (std::runtime_error &error) {
     std::cout << error.what() << std::endl;
     ResponseErrorMessage errorMessage{ std::string{error.what()} };
     return std::make_pair(Status_Error, errorMessage.getBufferData());
  }

  interpreter::DMLResponseMessage responsePayload{resultToken};
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
