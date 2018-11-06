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
#include <cuda_runtime.h>

#include "gdf/gdf.h"

#include <tuple>
#include <blazingdb/protocol/api.h>
#include <blazingdb/protocol/message/messages.h>
#include <blazingdb/protocol/message/interpreter/messages.h>
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

static result_pair getResultService(uint64_t accessToken, Buffer&& requestPayloadBuffer) {
  std::cout << "accessToken: " << accessToken << std::endl;

  interpreter::GetResultRequestMessage request(requestPayloadBuffer.data());
  std::cout << "resultToken: " << request.getResultToken() << std::endl;


  // remove from repository using accessToken and resultToken
  blazing_frame result = result_set_repository::get_instance().get_result(accessToken, request.getResultToken());

  //TODO ojo el result siempre es una sola tabla por eso indice 0
  const int rows = result.get_columns()[0][0].size();

  interpreter::BlazingMetadataDTO  metadata = {
    .status = "OK",
    .message = "metadata message",
    .time = 0.1f,
    .rows = rows
  };

  std::vector<std::string> fieldNames;
  std::vector<::gdf_dto::gdf_column> values;

  //TODO WARNING why 0 why multitables?
  for(int i = 0; i < result.get_columns()[0].size(); ++i) {
	  fieldNames.push_back(result.get_columns()[0][i].name());

	  auto data = libgdf::BuildCudaIpcMemHandler(result.get_columns()[0][i].get_gdf_column()->data);
	  auto valid = libgdf::BuildCudaIpcMemHandler(result.get_columns()[0][i].get_gdf_column()->valid);

	  auto col = ::gdf_dto::gdf_column {
	        .data = data,
	        .valid = valid,
	        .size = result.get_columns()[0][i].size(),
	        .dtype = (gdf_dto::gdf_dtype)result.get_columns()[0][i].dtype(),
	        .null_count = result.get_columns()[0][i].null_count(),
	        .dtype_info = gdf_dto::gdf_dtype_extra_info {
	          .time_unit = (gdf_dto::gdf_time_unit)0,
	        }
	    };

	  values.push_back(col);
  }

//  // todo: remove hardcode by creating the resulset vector
//  gdf_column_cpp column = result.get_columns()[0][0];
//	std::cout<<"getResultService\n";
//  print_gdf_column(column.get_gdf_column());
//  std::cout<<"end:getResultService\n";
//
//  auto data = libgdf::BuildCudaIpcMemHandler(column.get_gdf_column()->data);
//  auto valid = libgdf::BuildCudaIpcMemHandler(column.get_gdf_column()->valid);
//
//  std::vector<::gdf_dto::gdf_column> values = {
//    ::gdf_dto::gdf_column {
//        .data = data,
//        .valid = valid,
//        .size = column.size(),
//        .dtype = (gdf_dto::gdf_dtype)column.dtype(),
//        .null_count = column.null_count(),
//        .dtype_info = gdf_dto::gdf_dtype_extra_info {
//          .time_unit = (gdf_dto::gdf_time_unit)0,
//        }
//    }
//  };

  interpreter::GetResultResponseMessage responsePayload(metadata, fieldNames, values);
  std::cout << "**before return data frame\n" << std::flush;
  return std::make_pair(Status_Success, responsePayload.getBufferData());
}

static result_pair freeResultService(uint64_t accessToken, Buffer&& requestPayloadBuffer) {
   std::cout << "freeResultService: " << accessToken << std::endl;

  interpreter::GetResultRequestMessage request(requestPayloadBuffer.data());
  std::cout << "resultToken: " << request.getResultToken() << std::endl;
  if(result_set_repository::get_instance().free_result(request.getResultToken())){
	  ZeroMessage response{};
	  return std::make_pair(Status_Success, response.getBufferData());
  }else{
	  ResponseErrorMessage errorMessage{ std::string{"Could not free result set!"} };
	  return std::make_pair(Status_Error, errorMessage.getBufferData());
  }

}

static result_pair executePlanService(uint64_t accessToken, Buffer&& requestPayloadBuffer)   {
  interpreter::ExecutePlanRequestMessage requestPayload(requestPayloadBuffer.data());

  // ExecutePlan
  std::cout << "accessToken: " << accessToken << std::endl;
  std::cout << "query: " << requestPayload.getLogicalPlan() << std::endl;
  std::cout << "tableGroup: " << requestPayload.getTableGroup().name << std::endl;
 	std::cout << "tables: " << requestPayload.getTableGroup().tables.size() << std::endl;
  std::cout << "tableSize: " << requestPayload.getTableGroup().tables.size() << std::endl;
	std::cout << "FirstColumnSize: "
			<< requestPayload.getTableGroup().tables[0].columns[0].size
			<< std::endl;
	  std::vector<void *> handles;
	std::tuple<std::vector<std::vector<gdf_column_cpp>>, std::vector<std::string>, std::vector<std::vector<std::string>>> request = libgdf::toBlazingDataframe(requestPayload.getTableGroup(),handles);

  uint64_t resultToken = 0L;
  try {
    resultToken = evaluate_query(std::get<0>(request), std::get<1>(request), std::get<2>(request),
                                        requestPayload.getLogicalPlan(), accessToken,handles);
  } catch (std::exception& error) {
     std::cout << error.what() << std::endl;
     ResponseErrorMessage errorMessage{ std::string{error.what()} };
     return std::make_pair(Status_Error, errorMessage.getBufferData());
  }
  interpreter::NodeConnectionDTO nodeInfo {
      .path = "/tmp/ral.socket",
      .type = NodeConnectionType {NodeConnectionType_IPC}
  };
  interpreter::ExecutePlanResponseMessage responsePayload{resultToken, nodeInfo};
  return std::make_pair(Status_Success, responsePayload.getBufferData());
}

int main(void)
{
	std::cout << "RAL Engine starting"<< std::endl;

  blazingdb::protocol::UnixSocketConnection connection({"/tmp/ral.socket", std::allocator<char>()});
  blazingdb::protocol::Server server(connection);

  std::map<int8_t, FunctionType> services;
  services.insert(std::make_pair(interpreter::MessageType_ExecutePlan, &executePlanService));
  services.insert(std::make_pair(interpreter::MessageType_CloseConnection, &closeConnectionService));
  services.insert(std::make_pair(interpreter::MessageType_GetResult, &getResultService));
  services.insert(std::make_pair(interpreter::MessageType_FreeResult, &freeResultService));

  auto interpreterServices = [&services](const blazingdb::protocol::Buffer &requestPayloadBuffer) -> blazingdb::protocol::Buffer {
    RequestMessage request{requestPayloadBuffer.data()};
    std::cout << "header: " << (int)request.messageType() << std::endl;

    auto result = services[request.messageType()] ( request.accessToken(),  request.getPayloadBuffer() );
    ResponseMessage responseObject{result.first, result.second};
    return Buffer{responseObject.getBufferData()};
  };
  server.handle(interpreterServices);

	return 0;
}
