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
#include "DataFrame.h"
#include "Utils.cuh"
#include "Types.h"

#include "ipc/calcite_client.h"
#include "gdf/gdf.h"

#include <blazingdb/protocol/api.h>
#include <blazingdb/protocol/interpreter/messages.h>

namespace blazingdb {
	namespace protocol {
		namespace interpreter {
			auto InterpreterService(const blazingdb::protocol::Buffer &requestBuffer) -> blazingdb::protocol::Buffer {
				RequestMessage request{requestBuffer.data()};
				DMLRequestMessage requestPayload(request.getPayloadBuffer());

				std::vector<std::vector<gdf_column_cpp> > input_tables;
				std::vector<std::string> table_names;
				std::vector<std::vector<std::string>> column_names;
				std::string logicalPlan = requestPayload.getLogicalPlan();
				connection_id_t connection = request.accessToken();

				query_token_t token = evaluate_query(input_tables, table_names, column_names,
												logicalPlan, connection);

				DMLResponseMessage responsePayload{token};
				ResponseMessage responseObject{Status_Success, responsePayload};
				auto bufferedData = responseObject.getBufferData();
				Buffer buffer{bufferedData->data(),
							bufferedData->size()};
				return buffer;
			}
		}
	}
}

/*namespace blazingdb {
	namespace protocol {
	  namespace interpreter {
		auto InterpreterService(const blazingdb::protocol::Buffer &requestBuffer) -> blazingdb::protocol::Buffer {
			RequestMessage request{requestBuffer.data()};
			DMLRequestMessage requestPayload(request.getPayloadBuffer());

			int_t status=Status_Success;

			try {
				result_set_repository::get_instance().remove_all_connection_tokens(requestPayload.getAccessToken());
				status = Status_Success;
			}
			catch(const std::runtime_error& e) {
				std::cout<<"Error: "<<e.what()<<std::endl;
				status = Status_Error;
				strcpy(resp.msg, e.what());
			}

			ResponseErrorMessage responsePayload{error};
			ResponseMessage responseObject{status, responsePayload};

			auto bufferedData = responseObject.getBufferData();
			Buffer buffer{bufferedData->data(),
						  bufferedData->size()};
			return buffer;
		}
	  }
	}
}*/

int main(void)
{
	using namespace blazingdb::protocol::interpreter;

	blazingdb::protocol::UnixSocketConnection connection({"/tmp/ral.socket", std::allocator<char>()});
	blazingdb::protocol::Server server(connection);

	server.handle(InterpreterService);

	return 0;
}
