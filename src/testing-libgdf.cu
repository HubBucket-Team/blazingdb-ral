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
#include <memory>
#include <algorithm>
#include <thread>
#include "CalciteInterpreter.h"
#include "ResultSetRepository.h"
#include "DataFrame.h"
#include "Utils.cuh"
#include "Types.h"
#include <cuda_runtime.h>

#include "FreeMemory.h"

#include "gdf_wrapper/gdf_wrapper.cuh"

#include <tuple>

#include <blazingdb/protocol/api.h>
#include <blazingdb/protocol/message/messages.h>
#include <blazingdb/protocol/message/orchestrator/messages.h>
#include <blazingdb/protocol/message/interpreter/messages.h>
#include <blazingdb/protocol/message/io/file_system.h>
#include "ral-message.cuh"


using namespace blazingdb::protocol;

#include <blazingdb/io/Util/StringUtil.h>

#include <blazingdb/io/FileSystem/HadoopFileSystem.h>
#include <blazingdb/io/FileSystem/S3FileSystem.h>
#include <blazingdb/io/FileSystem/FileSystemRepository.h>
#include <blazingdb/io/FileSystem/FileSystemCommandParser.h>
#include <blazingdb/io/FileSystem/FileSystemManager.h>
#include <blazingdb/io/Config/BlazingContext.h>
#include <blazingdb/io/Library/Logging/Logger.h>
#include <blazingdb/io/Library/Logging/CoutOutput.h>
#include <blazingdb/io/Library/Logging/FileOutput.h>
#include "blazingdb/io/Library/Logging/ServiceLogging.h"

#include "CalciteExpressionParsing.h"
#include "io/data_parser/CSVParser.h"
#include "io/data_parser/GDFParser.h"
#include "io/data_parser/ParquetParser.h"

#include "io/data_provider/UriDataProvider.h"
#include "io/data_parser/DataParser.h"
#include "io/data_provider/DataProvider.h"
#include "io/DataLoader.h"


#include "Config/Config.h"

#include "CodeTimer.h"

const Path FS_NAMESPACES_FILE("/tmp/file_system.bin");
using result_pair = std::pair<Status, std::shared_ptr<flatbuffers::DetachedBuffer>>;
using FunctionType = result_pair (*)(uint64_t, Buffer&& buffer);

//TODO percy c.gonzales fix this later
std::string global_ip;
int global_port;

static result_pair  registerFileSystem(uint64_t accessToken, Buffer&& buffer) {
  std::cout << "registerFileSystem: " << accessToken << std::endl;
  blazingdb::message::io::FileSystemRegisterRequestMessage message(buffer.data());

  FileSystemConnection fileSystemConnection;
  Path root("/");
  const std::string authority =  message.getAuthority();
  if (message.isLocal()) {
    fileSystemConnection = FileSystemConnection(FileSystemType::LOCAL);
  } else if (message.isHdfs()) {
    auto hdfs = message.getHdfs();
    fileSystemConnection = FileSystemConnection(hdfs.host, hdfs.port, hdfs.user, (HadoopFileSystemConnection::DriverType)hdfs.driverType, hdfs.kerberosTicket);
  } else if (message.isS3()) {
    auto s3 = message.getS3();
    fileSystemConnection = FileSystemConnection(s3.bucketName, ( S3FileSystemConnection::EncryptionType )s3.encryptionType, s3.kmsKeyAmazonResourceName, s3.accessKeyId, s3.secretKey, s3.sessionToken);
  }
  root = message.getRoot();
  if (root.isValid() == false) {
    std::cout << "something went wrong when registering filesystem ..." << std::endl;
    ResponseErrorMessage errorMessage{ std::string{ "ERROR: Invalid root provided when registering file system"} };
    return std::make_pair(Status_Error, errorMessage.getBufferData());
  }
  FileSystemEntity fileSystemEntity(authority, fileSystemConnection, root);
  bool ok = BlazingContext::getInstance()->getFileSystemManager()->deregisterFileSystem(authority);
  ok = BlazingContext::getInstance()->getFileSystemManager()->registerFileSystem(fileSystemEntity);
	if (ok) { // then save the fs
		const FileSystemRepository fileSystemRepository(FS_NAMESPACES_FILE, true);
		const bool saved = fileSystemRepository.add(fileSystemEntity);
		if (saved == false) {
			std::cout << "WARNING: could not save the registered file system into ... the data file uri ..."; //TODO percy error message
		}
	} else {
   	  std::cout << "something went wrong when registering filesystem ..." << std::endl;
      ResponseErrorMessage errorMessage{ std::string{"ERROR: Something went wrong when registering file system"} };
      return std::make_pair(Status_Error, errorMessage.getBufferData());
	}
  ZeroMessage response{};
  return std::make_pair(Status_Success, response.getBufferData());
}

static result_pair  deregisterFileSystem(uint64_t accessToken, Buffer&& buffer)  {
  std::cout << "deregisterFileSystem: " << accessToken << std::endl;
  blazingdb::message::io::FileSystemDeregisterRequestMessage message(buffer.data());
  auto authority =  message.getAuthority();
  if (authority.empty() == true) {
     ResponseErrorMessage errorMessage{ std::string{"derigistering an empty authority"} };
     return std::make_pair(Status_Error, errorMessage.getBufferData());
  }
  const bool ok = BlazingContext::getInstance()->getFileSystemManager()->deregisterFileSystem(authority);
  if (ok) { // then save the fs
    const FileSystemRepository fileSystemRepository(FS_NAMESPACES_FILE, true);
    const bool deleted = fileSystemRepository.deleteByAuthority(authority);
    if (deleted == false) {
      std::cout << "WARNING: could not delete the registered file system into ... the data file uri ..."; //TODO percy error message
    }
  }
  ZeroMessage response{};
  return std::make_pair(Status_Success, response.getBufferData());
}


using result_pair = std::pair<Status, std::shared_ptr<flatbuffers::DetachedBuffer>>;
using FunctionType = result_pair (*)(uint64_t, Buffer&& buffer);

static result_pair closeConnectionService(uint64_t accessToken, Buffer&& requestPayloadBuffer) {
  std::cout << "accessToken: " << accessToken << std::endl;

  try {
    result_set_repository::get_instance().remove_all_connection_tokens(accessToken);
    // NOTE: use next 3 lines to check with "/usr/local/cuda/bin/cuda-memcheck  --leak-check full  ./testing-libgdf"   
    // GDFRefCounter::getInstance()->show_summary();
    // cudaDeviceReset();
    // exit(0);
  } catch (const std::exception& e) {
     std::cerr << e.what() << std::endl;
     ResponseErrorMessage errorMessage{ std::string{e.what()} };
     return std::make_pair(Status_Error, errorMessage.getBufferData());
  }

  ZeroMessage response{};
  return std::make_pair(Status_Success, response.getBufferData());
}

static result_pair getResultService(uint64_t accessToken, Buffer&& requestPayloadBuffer) {
  std::cout << "accessToken: " << accessToken << std::endl;

  interpreter::GetResultRequestMessage request(requestPayloadBuffer.data());
  std::cout << "resultToken: " << request.getResultToken() << std::endl;

  try {
    // get result from repository using accessToken and resultToken
    result_set_t result = result_set_repository::get_instance().get_result(accessToken, request.getResultToken());

    
    std::string status = "Error";
    std::string errorMsg = result.errorMsg;
    std::vector<std::string> fieldNames;
    std::vector<uint64_t> columnTokens;
    std::vector<::gdf_dto::gdf_column> values;
    int rows = 0;

    if (errorMsg.empty()) {
      status = "OK";
      //TODO ojo el result siempre es una sola tabla por eso indice 0
      rows =  result.result_frame.get_columns()[0][0].size();


      for(std::size_t i = 0; i < result.result_frame.get_columns()[0].size(); ++i) {
        fieldNames.push_back(result.result_frame.get_columns()[0][i].name());
        columnTokens.push_back(result.result_frame.get_columns()[0][i].get_column_token());

        std::cout << "col_name: " << result.result_frame.get_columns()[0][i].name() << std::endl;
        nvstrings_ipc_transfer ipc;
        gdf_dto::gdf_dtype_extra_info dtype_info;
        ::gdf_dto::gdf_column col;

        std::basic_string<int8_t> data;
        std::basic_string<int8_t> valid;

        if(result.result_frame.get_columns()[0][i].dtype() == GDF_STRING){
          NVStrings* strings = static_cast<NVStrings *> (result.result_frame.get_columns()[0][i].get_gdf_column()->data);
          if(result.result_frame.get_columns()[0][i].size() > 0)
            strings->create_ipc_transfer(ipc);
          dtype_info = gdf_dto::gdf_dtype_extra_info {
                .time_unit = (gdf_dto::gdf_time_unit)0,
            };
          // before
          /*  col = ::gdf_dto::gdf_column {
              .data = data,
              .valid = valid,
              .size = static_cast<gdf_size_type>(result.result_frame.get_columns()[0][i].size()),//.get_gdf_column()->data  ` 
              .dtype = (gdf_dto::gdf_dtype)result.result_frame.get_columns()[0][i].dtype(), // GDF_STRING
              .null_count = static_cast<gdf_size_type>(result.result_frame.get_columns()[0][i].null_count()),
              .dtype_info = dtype_info,
              // custrings data
              .custrings_data = libgdf::ConvertIpcByteArray(ipc)
          };
          */
          col.data = data;
          col.valid = valid;
          col.size = result.result_frame.get_columns()[0][i].size();
          col.dtype =  (gdf_dto::gdf_dtype)result.result_frame.get_columns()[0][i].dtype();
          col.dtype_info = dtype_info;
          col.null_count = static_cast<gdf_size_type>(result.result_frame.get_columns()[0][i].null_count()),
          // custrings data
          col.custrings_data = libgdf::ConvertIpcByteArray(ipc);

        }else{
          dtype_info = gdf_dto::gdf_dtype_extra_info {
                .time_unit = (gdf_dto::gdf_time_unit)0     // TODO: why is this hardcoded?
          };

          data = libgdf::BuildCudaIpcMemHandler(result.result_frame.get_columns()[0][i].get_gdf_column()->data);
          valid = libgdf::BuildCudaIpcMemHandler(result.result_frame.get_columns()[0][i].get_gdf_column()->valid);
        
        // before
        /*  col = ::gdf_dto::gdf_column {
              .data = data,
              .valid = valid,
              .size = result.result_frame.get_columns()[0][i].size(),
              .dtype = (gdf_dto::gdf_dtype)result.result_frame.get_columns()[0][i].dtype(), 
              .null_count = result.result_frame.get_columns()[0][i].null_count(),
              .dtype_info = dtype_info
          };
        */
          col.data = data;
          col.valid = valid;
          col.size = result.result_frame.get_columns()[0][i].size();
          col.dtype =  (gdf_dto::gdf_dtype)result.result_frame.get_columns()[0][i].dtype();
          col.null_count = result.result_frame.get_columns()[0][i].null_count();
          col.dtype_info = dtype_info;

        }

        values.push_back(col);
      }
    }   

    interpreter::BlazingMetadataDTO  metadata = {
      .status = status,
      .message = errorMsg,
      .time = result.duration,
      .rows = rows
    };

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

    interpreter::GetResultResponseMessage responsePayload(metadata, fieldNames, columnTokens, values);
    return std::make_pair(Status_Success, responsePayload.getBufferData());

  } catch (const std::exception& e) {
     std::cerr << e.what() << std::endl;
     ResponseErrorMessage errorMessage{ std::string{e.what()} };
     return std::make_pair(Status_Error, errorMessage.getBufferData());
  } catch (...) {
    ResponseErrorMessage errorMessage{ std::string{"Unknown error"} };
    return std::make_pair(Status_Error, errorMessage.getBufferData());
  }
}

static result_pair freeResultService(uint64_t accessToken, Buffer&& requestPayloadBuffer) {
   std::cout << "freeResultService: " << accessToken << std::endl;

  interpreter::GetResultRequestMessage request(requestPayloadBuffer.data());
  std::cout << "resultToken: " << request.getResultToken() << std::endl;
  bool success = false;
  try {
    success = result_set_repository::get_instance().try_free_result(accessToken, request.getResultToken());
  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
     ResponseErrorMessage errorMessage{ std::string{e.what()} };
     return std::make_pair(Status_Error, errorMessage.getBufferData());
  }
  if(success){
	  ZeroMessage response{};
	  return std::make_pair(Status_Success, response.getBufferData());
  }else{
	  ResponseErrorMessage errorMessage{ std::string{"Could not free result set!"} };
	  return std::make_pair(Status_Error, errorMessage.getBufferData());
  }

}


//TODO: we need to have a centralized place where this can be done
//perhaps a utility in protocol, add bool8 after update
gdf_dtype convert_string_dtype(std::string str){
	if(str == "GDF_INT8"){
		return GDF_INT8;
	}else if(str == "GDF_INT16"){
		return GDF_INT16;
	}else if(str == "GDF_INT32"){
		return GDF_INT32;
	}else if(str == "GDF_INT64"){
		return GDF_INT64;
	}else if(str == "GDF_FLOAT32"){
		return GDF_FLOAT32;
	}else if(str == "GDF_FLOAT64"){
		return GDF_FLOAT64;
	}else if(str == "GDF_DATE32"){
		return GDF_DATE32;
	}else if(str == "GDF_DATE64"){
		return GDF_DATE64;
	}else if(str == "GDF_TIMESTAMP"){
		return GDF_TIMESTAMP;
	}else if(str == "GDF_CATEGORY"){
		return GDF_CATEGORY;
	}else if(str == "GDF_STRING"){
		return GDF_STRING;
	}else if(str == "GDF_STRING_CATEGORY"){
		return GDF_STRING_CATEGORY;
	}else{
		return GDF_INVALID;
	}
}


static result_pair parseSchemaService(uint64_t accessToken, Buffer&& requestPayloadBuffer) {
	blazingdb::protocol::orchestrator::DDLCreateTableRequestMessage requestPayload(requestPayloadBuffer.data());

	std::shared_ptr<ral::io::data_parser> parser;
	if(requestPayload.schemaType == blazingdb::protocol::FileSchemaType::FileSchemaType_PARQUET){
		parser = std::make_shared<ral::io::parquet_parser>();

	}else if(requestPayload.schemaType == blazingdb::protocol::FileSchemaType::FileSchemaType_CSV){
		std::vector<gdf_dtype> types;
		for(auto val : requestPayload.columnTypes){
			types.push_back(convert_string_dtype(val));
		}
		parser =  std::make_shared<ral::io::csv_parser>(
				requestPayload.csvDelimiter,
  				requestPayload.csvLineTerminator,
  				(int) requestPayload.csvSkipRows,
  				requestPayload.columnNames, types);
	}else{
		//indicate error here
		//this shoudl be done in the orchestrator
	}

	 std::vector<Uri> uris;
	 for (auto file_path : requestPayload.files) {
	     uris.push_back(Uri{file_path});
	 }

	auto provider = std::make_shared<ral::io::uri_data_provider>(uris);
	auto loader = std::make_shared<ral::io::data_loader>( parser,provider);
	ral::io::Schema schema;
	loader->get_schema(schema);


	blazingdb::protocol::TableSchemaSTL transport_schema = schema.getTransport();

	if(requestPayload.schemaType == blazingdb::protocol::FileSchemaType::FileSchemaType_CSV){
		transport_schema.csvDelimiter = requestPayload.csvDelimiter;
		transport_schema.csvSkipRows = requestPayload.csvSkipRows;
		transport_schema.csvLineTerminator = requestPayload.csvLineTerminator;
	}
	transport_schema.files = requestPayload.files;

	blazingdb::protocol::interpreter::CreateTableResponseMessage responsePayload(transport_schema);
	return std::make_pair(Status_Success, responsePayload.getBufferData());
}

static result_pair executeFileSystemPlanService (uint64_t accessToken, Buffer&& requestPayloadBuffer) {
  blazingdb::message::io::FileSystemDMLRequestMessage requestPayload(requestPayloadBuffer.data());

  //make dataloaders
	std::vector<ral::io::data_loader > input_loaders;
	std::vector<ral::io::Schema> schemas;
	std::vector<std::string> table_names;
  for(auto table : requestPayload.tableGroup.tables){
	  ral::io::Schema schema(table.tableSchema);
	std::shared_ptr<ral::io::data_parser> parser;
	  if(table.schemaType == blazingdb::protocol::FileSchemaType::FileSchemaType_PARQUET){
	  		parser = std::make_shared<ral::io::parquet_parser>();

	  	}else if(table.schemaType == blazingdb::protocol::FileSchemaType::FileSchemaType_CSV){
	  		std::vector<gdf_dtype> types;
			for(auto val : table.tableSchema.types){
				types.push_back((gdf_dtype) val);
			}

	  		parser =  std::make_shared<ral::io::csv_parser>(
	  				table.tableSchema.csvDelimiter,
	  				table.tableSchema.csvLineTerminator,
	  				table.tableSchema.csvSkipRows,
	  				table.tableSchema.names, types);
	  	}else{
	  		parser = std::make_shared<ral::io::gdf_parser>(table);
	  	}


	  std::shared_ptr<ral::io::data_provider> provider;
	  std::vector<Uri> uris;
	  	 for (auto file_path : schema.files) {
	  	     uris.push_back(Uri{file_path});
	  	 }



	  if(table.schemaType == blazingdb::protocol::FileSchemaType::FileSchemaType_CSV ||
			  table.schemaType == blazingdb::protocol::FileSchemaType::FileSchemaType_PARQUET){
		  	 provider = std::make_shared<ral::io::uri_data_provider>(uris);
	  }else{
		  provider = std::make_shared<ral::io::dummy_data_provider>();
	  }
	  	auto loader = std::make_shared<ral::io::data_loader>( parser,provider);
	  	input_loaders.push_back(loader);
	  	schemas.push_back(schema);
	  	table_names.push_back(table.name);

  }


  std::cout << "accessToken: " << accessToken << std::endl;
  std::cout << "query: " << requestPayload.statement << std::endl;
  std::cout << "tableGroup: " << requestPayload.tableGroup.name << std::endl;
 	std::cout << "tables: " << requestPayload.tableGroup.tables.size() << std::endl;
  std::cout << "tableSize: " << requestPayload.tableGroup.tables.size() << std::endl;
	std::cout << "FirstColumn File: "
			<< requestPayload.tableGroup.tables[0].files[0]
			<< std::endl;
  
  uint64_t resultToken = 0L;
  try {


    // Execute query
    resultToken = evaluate_query(input_loaders, schemas, table_names, requestPayload.statement, accessToken );
  } catch (const std::exception& e) {
     std::cerr << e.what() << std::endl;
     ResponseErrorMessage errorMessage{ std::string{e.what()} };
     return std::make_pair(Status_Error, errorMessage.getBufferData());
  }

  interpreter::NodeConnectionDTO nodeInfo {
      .port = global_port,
      .path = "/tmp/ral.socket",
      .type = NodeConnectionType {NodeConnectionType_TCP}
  };
  interpreter::ExecutePlanResponseMessage responsePayload{resultToken, nodeInfo};
  return std::make_pair(Status_Success, responsePayload.getBufferData());
}

static result_pair freeMemoryCallback(uint64_t accessToken, Buffer&& requesBuffer)   {
    FreeMemory::freeAll();
    ZeroMessage response{};
    return std::make_pair(Status_Success, response.getBufferData());
}

static  std::map<int8_t, FunctionType> services;


//@todo execuplan with filesystem
auto  interpreterServices(const blazingdb::protocol::Buffer &requestPayloadBuffer) -> blazingdb::protocol::Buffer {
  RequestMessage request{requestPayloadBuffer.data()};
  std::cout << "header: " << (int)request.messageType() << std::endl;

  auto result = services[request.messageType()] ( request.accessToken(),  request.getPayloadBuffer() );
  ResponseMessage responseObject{result.first, result.second};
  return Buffer{responseObject.getBufferData()};
}


int main(int argc, const char *argv[])
{

  /*#ifndef VERBOSE
  std::cout.rdbuf(nullptr); // substitute internal std::cout buffer with
  #endif // VERBOSE*/
  
    std::cout << "RAL Engine starting" << std::endl;


    FreeMemory::Initialize();

    auto output = new Library::Logging::FileOutput("RAL.log", true);
    Library::Logging::ServiceLogging::getInstance().setLogOutput(output);

    // Init AWS S3 ... TODO see if we need to call shutdown and avoid leaks from s3 percy
    BlazingContext::getInstance()->initExternalSystems();
    
  global_ip = "/tmp/ral.socket";
  //global_port = atoi(port.c_str());

  blazingdb::protocol::UnixSocketConnection connection("/tmp/ral.socket");
  blazingdb::protocol::Server server(connection);

  services.insert(std::make_pair(interpreter::MessageType_ExecutePlanFileSystem, &executeFileSystemPlanService));

  services.insert(std::make_pair(interpreter::MessageType_CloseConnection, &closeConnectionService));
  services.insert(std::make_pair(interpreter::MessageType_GetResult, &getResultService));
  services.insert(std::make_pair(interpreter::MessageType_FreeResult, &freeResultService));
  services.insert(std::make_pair(interpreter::MessageType_RegisterFileSystem, &registerFileSystem));
  services.insert(std::make_pair(interpreter::MessageType_DeregisterFileSystem, &deregisterFileSystem));

  services.insert(std::make_pair(9, &freeMemoryCallback));

  server.handle(&interpreterServices);

	return 0;
}
