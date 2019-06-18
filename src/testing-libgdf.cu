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
#include "io/data_provider/DummyProvider.h"
#include "io/data_provider/UriDataProvider.h"

#include "io/data_parser/DataParser.h"
#include "io/data_provider/DataProvider.h"
#include "io/DataLoader.h"


#include "CodeTimer.h"
#include "config/BlazingConfig.h"

#include "communication/CommunicationData.h"
#include "communication/factory/MessageFactory.h"
#include "communication/network/Client.h"
#include "communication/network/Server.h"
#include <blazingdb/communication/Context.h>

const Path FS_NAMESPACES_FILE("/tmp/file_system.bin");
using result_pair = std::pair<Status, std::shared_ptr<flatbuffers::DetachedBuffer>>;
using FunctionType = result_pair (*)(uint64_t, Buffer&& buffer);

ConnectionAddress connectionAddress;

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


query_token_t loadParquetAndInsertToResultRepository(std::string path, connection_id_t connection, bool schema_only) {
	std::cout<<"loadParquet\n";

	query_token_t token = result_set_repository::get_instance().register_query(connection); //register the query so we can receive result requests for it
  Path  csvFile(path);
	Uri directory(csvFile.getParentPath().toString());
  const bool existsParquetDir = BlazingContext::getInstance()->getFileSystemManager()->exists(directory);
	const bool isS3Dir = (directory.getFileSystemType() == FileSystemType::S3);
	if ((existsParquetDir == false) && (isS3Dir == false)) {
    auto error = "parquet table folder doesn't exists: " + directory.toString(false);
    throw std::runtime_error{error};
  }

	std::thread t = std::thread([=]{
		std::vector<Uri> uris(1);
		uris[0] = Uri(path);
		
        auto provider = ral::io::uri_data_provider(uris);
        auto parser = ral::io::parquet_parser();
      ral::io::data_loader loader(&parser, &provider);
      
	  try
	  {
	  
	    CodeTimer blazing_timer;
	    std::vector<gdf_column_cpp> columns;
	    loader.load_data(columns, {}, false);

      blazing_frame output_frame;
      output_frame.add_table(columns);

      double duration = blazing_timer.getDuration();

      result_set_repository::get_instance().update_token(token, output_frame, duration);
    } catch(const std::exception& e) {
      std::cerr << e.what() << '\n';
      result_set_repository::get_instance().update_token(token, blazing_frame{}, 0.0, e.what());
    }
	});
	t.detach();
	return token;
}

static result_pair loadParquetSchema(uint64_t accessToken, Buffer&& buffer) {

  blazingdb::message::io::LoadParquetFileRequestMessage message(buffer.data());

  uint64_t resultToken = 0L;
  try {
    // @todo, what about other parameters
    bool schema_only = true;
    resultToken = loadParquetAndInsertToResultRepository(message.fileSchema()->path, accessToken, schema_only);

  } catch (const std::exception& e) {
     std::cerr << e.what() << std::endl;
     ResponseErrorMessage errorMessage{ std::string{e.what()} };
     return std::make_pair(Status_Error, errorMessage.getBufferData());
  }
  
  #ifdef USE_UNIX_SOCKETS

  interpreter::NodeConnectionDTO nodeInfo {
      .port = -1,
      .path = ral::config::BlazingConfig::getInstance().getSocketPath(),
      .type = NodeConnectionType {NodeConnectionType_TCP}
  };

  #else

  interpreter::NodeConnectionDTO nodeInfo {
      .port = connectionAddress.tcp_port,
      .path = ral::config::BlazingConfig::getInstance().getSocketPath(),
      .type = NodeConnectionType {NodeConnectionType_TCP}
  };

  #endif
  
  interpreter::ExecutePlanResponseMessage responsePayload{resultToken, nodeInfo};
  return std::make_pair(Status_Success, responsePayload.getBufferData());
}

query_token_t loadCsvAndInsertToResultRepository(std::string path, std::vector<std::string> names, std::vector<gdf_dtype> dtypes, std::string delimiter, std::string line_terminator, 
  int skip_rows, connection_id_t connection, bool schema_only) {
	std::cout<<"loadCsv\n";

	query_token_t token = result_set_repository::get_instance().register_query(connection); //register the query so we can receive result requests for it
  Path  csvFile(path);
	Uri directory(csvFile.getParentPath().toString());
  const bool existsParquetDir = BlazingContext::getInstance()->getFileSystemManager()->exists(directory);
	const bool isS3Dir = (directory.getFileSystemType() == FileSystemType::S3);
	if ((existsParquetDir == false) && (isS3Dir == false)) {
    auto error = "csv table folder doesn't exists: " + directory.toString(false);
    throw std::runtime_error{error};
  }
	std::thread t = std::thread([=]{
		std::vector<Uri> uris(1);
		uris[0] = Uri(path);

		auto provider = ral::io::uri_data_provider(uris);
		auto parser = ral::io::csv_parser(delimiter, line_terminator, skip_rows, names, dtypes);
	  ral::io::data_loader loader(&parser, &provider);

    try
    {
      CodeTimer blazing_timer;

      size_t num_cols = names.size();
      std::vector<bool> include_column(num_cols, true);


      std::vector<gdf_column_cpp> columns;
      loader.load_data(columns, {}, false);

      blazing_frame output_frame;
      output_frame.add_table(columns);

      double duration = blazing_timer.getDuration();

      result_set_repository::get_instance().update_token(token, output_frame, duration);
    } catch (const std::exception& e){
      std::cerr << e.what() << '\n';
      result_set_repository::get_instance().update_token(token, blazing_frame{}, 0.0, e.what());
    }
	});
	t.detach();
	return token;
}

static result_pair loadCsvSchema(uint64_t accessToken, Buffer&& buffer) {
  blazingdb::message::io::LoadCsvFileRequestMessage message(buffer.data());

  std::vector<gdf_dtype> types;
  auto schema = message.fileSchema();
  for(auto val : schema->dtypes)
    types.push_back( (gdf_dtype) val );

  uint64_t resultToken = 0L;
  try {
    bool schema_only = true;
    resultToken = loadCsvAndInsertToResultRepository(schema->path, schema->names, types, schema->delimiter, schema->line_terminator, schema->skip_rows, accessToken, schema_only);
  } catch (const std::exception& e) {
     std::cerr << e.what() << std::endl;
     ResponseErrorMessage errorMessage{ std::string{e.what()} };
     return std::make_pair(Status_Error, errorMessage.getBufferData());
  }
  
  #ifdef USE_UNIX_SOCKETS

  interpreter::NodeConnectionDTO nodeInfo {
      .port = -1,
      .path = ral::config::BlazingConfig::getInstance().getSocketPath(),
      .type = NodeConnectionType {NodeConnectionType_TCP}
  };

  #else

  interpreter::NodeConnectionDTO nodeInfo {
      .port = connectionAddress.tcp_port,
      .path = ral::config::BlazingConfig::getInstance().getSocketPath(),
      .type = NodeConnectionType {NodeConnectionType_TCP}
  };

  #endif

  interpreter::ExecutePlanResponseMessage responsePayload{resultToken, nodeInfo};
  return std::make_pair(Status_Success, responsePayload.getBufferData());
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
		return GDF_invalid;
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
  for(auto table : requestPayload.tableGroup().tables){
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
	  		parser = std::make_shared<ral::io::gdf_parser>(table,accessToken);
	  	}


	  std::shared_ptr<ral::io::data_provider> provider;
	  std::vector<Uri> uris;
	  	 for (auto file_path : table.tableSchema.files) {
	  	     uris.push_back(Uri{file_path});
	  	 }



	  if(table.schemaType == blazingdb::protocol::FileSchemaType::FileSchemaType_CSV ||
			  table.schemaType == blazingdb::protocol::FileSchemaType::FileSchemaType_PARQUET){
		  	 provider = std::make_shared<ral::io::uri_data_provider>(uris);
	  }else{
		  provider = std::make_shared<ral::io::dummy_data_provider>();
	  }
	  ral::io::data_loader loader( parser,provider);
	  	input_loaders.push_back(loader);
	  	schemas.push_back(schema);
	  	table_names.push_back(table.name);

  }


  std::cout << "accessToken: " << accessToken << std::endl;
  std::cout << "query: " << requestPayload.statement() << std::endl;
  std::cout << "tableGroup: " << requestPayload.tableGroup().name << std::endl;
 	std::cout << "num tables: " << requestPayload.tableGroup().tables.size() << std::endl;
  std::cout << "contextToken: " << requestPayload.communicationContext().token << std::endl;
  std::cout << "contextTotalNodes: " << requestPayload.communicationContext().nodes.size() << std::endl;
  
  uint64_t resultToken = 0L;
  try {

    using blazingdb::communication::ContextToken;
    using blazingdb::communication::Context;
    using blazingdb::communication::Node;
    using blazingdb::communication::Buffer;
    auto& rawCommContext = requestPayload.communicationContext();
    std::vector<std::shared_ptr<Node>> contextNodes;
    for(auto& rawNode: rawCommContext.nodes){
      auto& rawBuffer = rawNode.buffer;
      contextNodes.push_back(Node::Make(Buffer(reinterpret_cast<const char*>(rawBuffer.data()), rawBuffer.size())));
    }
    auto ctxToken = ContextToken::Make(rawCommContext.token);
    Context queryContext{ctxToken, contextNodes, contextNodes[rawCommContext.masterIndex], ""};
    ral::communication::network::Server::getInstance().registerContext(*ctxToken);


    // Execute query
    resultToken = evaluate_query(input_loaders, schemas, table_names, requestPayload.statement(), accessToken, queryContext );

  } catch (const std::exception& e) {
     std::cerr << e.what() << std::endl;
     ResponseErrorMessage errorMessage{ std::string{e.what()} };
     return std::make_pair(Status_Error, errorMessage.getBufferData());
  }

  #ifdef USE_UNIX_SOCKETS

  interpreter::NodeConnectionDTO nodeInfo {
      .port = -1,
      .path = ral::config::BlazingConfig::getInstance().getSocketPath(),
      .type = NodeConnectionType {NodeConnectionType_TCP}
  };

  #else

  interpreter::NodeConnectionDTO nodeInfo {
      .port = connectionAddress.tcp_port,
      .path = ral::config::BlazingConfig::getInstance().getSocketPath(),
      .type = NodeConnectionType {NodeConnectionType_TCP}
  };

  #endif

  interpreter::ExecutePlanResponseMessage responsePayload{resultToken, nodeInfo};
  return std::make_pair(Status_Success, responsePayload.getBufferData());
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
  std::cout << "resultToken: " << requestPayload.getTableGroup().tables[0].resultToken << std::endl;
  //Library::Logging::Logger().logInfo("query:\n" + requestPayload.getLogicalPlan());

  std::vector<void *> handles;
	uint64_t resultToken = 0L;
  try {
    std::tuple<std::vector<std::vector<gdf_column_cpp>>, std::vector<std::string>, std::vector<std::vector<std::string>>> request = libgdf::toBlazingDataframe(accessToken, requestPayload.getTableGroup(),handles);


    using blazingdb::communication::Context;
    using blazingdb::communication::Node;
    Context queryContext{std::vector<std::shared_ptr<Node>>{}, nullptr, ""};


    resultToken = evaluate_query(std::get<0>(request), std::get<1>(request), std::get<2>(request),
                                        requestPayload.getLogicalPlan(), accessToken, handles, queryContext);
  } catch (const std::exception& e) {
     std::cerr << e.what() << std::endl;
     ResponseErrorMessage errorMessage{ std::string{e.what()} };
     return std::make_pair(Status_Error, errorMessage.getBufferData());
  }
  
  #ifdef USE_UNIX_SOCKETS

  interpreter::NodeConnectionDTO nodeInfo {
      .port = -1,
      .path = ral::config::BlazingConfig::getInstance().getSocketPath(),
      .type = NodeConnectionType {NodeConnectionType_TCP}
  };

  #else

  interpreter::NodeConnectionDTO nodeInfo {
      .port = connectionAddress.tcp_port,
      .path = ral::config::BlazingConfig::getInstance().getSocketPath(),
      .type = NodeConnectionType {NodeConnectionType_TCP}
  };

  #endif

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

    std::cout << "Usage: " << argv[0]
            << " <RAL_ID>"
                " <ORCHESTRATOR_HTTP_COMMUNICATION_[IP|HOSTNAME]> <ORCHESTRATOR_HTTP_COMMUNICATION_PORT>"
                " <RAL_HTTP_COMMUNICATION_[IP|HOSTNAME]> <RAL_HTTP_COMMUNICATION_PORT> <RAL_TCP_PROTOCOL_PORT>" << std::endl;

    if (argc != 7) {
        std::cout << "FATAL: Invalid number of arguments" << std::endl;
        return EXIT_FAILURE;
    }

  // #ifndef VERBOSE
  // std::cout.rdbuf(nullptr); // substitute internal std::cout buffer with
  // #endif // VERBOSE

    std::cout << "RAL Engine starting" << std::endl;

    const std::string ralId = std::string(argv[1]);
    const std::string orchestratorHost = std::string(argv[2]);

    const int orchestratorCommunicationPort = ConnectionUtils::parsePort(argv[3]);

    if (orchestratorCommunicationPort == -1) {
        std::cout << "FATAL: Invalid Orchestrator HTTP communication port " + std::string(argv[3]) << std::endl;
        return EXIT_FAILURE;
    }
    
    const std::string ralHost = std::string(argv[4]);

    const int ralCommunicationPort = ConnectionUtils::parsePort(argv[5]);

    if (ralCommunicationPort == -1) {
        std::cout << "FATAL: Invalid RAL HTTP communication port " + std::string(argv[5]) << std::endl;
        return EXIT_FAILURE;
    }

    const int ralProtocolPort = ConnectionUtils::parsePort(argv[6]);

    if (ralProtocolPort == -1) {
        std::cout << "FATAL: Invalid RAL TCP protocol port " + std::string(argv[6]) << std::endl;
        return EXIT_FAILURE;
    }
    
    auto& communicationData = ral::communication::CommunicationData::getInstance();

    communicationData.initialize(
        std::atoi(ralId.c_str()),
        orchestratorHost,
        orchestratorCommunicationPort,
        ralHost,
        ralCommunicationPort,
        ralProtocolPort);

    std::cout << "RAL ID: " << ralId << std::endl;
    std::cout << "Orchestrator HTTP communication host: " << orchestratorHost << std::endl;
    std::cout << "Orchestrator HTTP communication port: " << orchestratorCommunicationPort << std::endl;
    std::cout << "RAL HTTP communication host: " << ralHost << std::endl;
    std::cout << "RAL HTTP communication port: " << ralCommunicationPort << std::endl;

    try {
        auto nodeDataMesssage = ral::communication::messages::Factory::createNodeDataMessage(communicationData.getSelfNode());
        ral::communication::network::Client::sendNodeData(communicationData.getOrchestratorIp(),
                                                          communicationData.getOrchestratorPort(),
                                                          nodeDataMesssage);

        ral::communication::network::Server::start(ralCommunicationPort);
    } catch (std::exception &e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }

    auto& config = ral::config::BlazingConfig::getInstance();


#ifdef USE_UNIX_SOCKETS

    config.setLogName("RAL." + ralId + ".log")
          .setSocketPath("/tmp/ral." + ralId + ".socket");

    std::cout << "Socket Name: " << config.getSocketPath() << std::endl;

#else

    // NOTE IMPORTANT PERCY aqui es que pyblazing se entera que este es el ip del RAL en el _send de pyblazing
    config.setLogName("RAL." + ralId + ".log")
          .setSocketPath(ralHost);

    std::cout << "Socket Name: " << config.getSocketPath() << std::endl;

#endif

    std::cout << "Log Name: " << config.getLogName() << std::endl;

    FreeMemory::Initialize();

    auto output = new Library::Logging::FileOutput(config.getLogName(), true);
    Library::Logging::ServiceLogging::getInstance().setLogOutput(output);

    // Init AWS S3 ... TODO see if we need to call shutdown and avoid leaks from s3 percy
    BlazingContext::getInstance()->initExternalSystems();

#ifdef USE_UNIX_SOCKETS

  connectionAddress.unix_socket_path = config.getSocketPath();
  blazingdb::protocol::UnixSocketConnection connection(connectionAddress);

#else

  connectionAddress.tcp_host = "127.0.0.1"; // NOTE always use localhost for protocol server
  connectionAddress.tcp_port = ralProtocolPort;

  std::cout << "RAL TCP protocol port: " << connectionAddress.tcp_port << std::endl;
  
#endif

  blazingdb::protocol::Server server(connectionAddress.tcp_port);

  services.insert(std::make_pair(interpreter::MessageType_ExecutePlanFileSystem, &executeFileSystemPlanService));
  services.insert(std::make_pair(interpreter::MessageType_LoadCsvSchema, &parseSchemaService));
  services.insert(std::make_pair(interpreter::MessageType_CloseConnection, &closeConnectionService));
  services.insert(std::make_pair(interpreter::MessageType_GetResult, &getResultService));
  services.insert(std::make_pair(interpreter::MessageType_FreeResult, &freeResultService));
  services.insert(std::make_pair(interpreter::MessageType_RegisterFileSystem, &registerFileSystem));
  services.insert(std::make_pair(interpreter::MessageType_DeregisterFileSystem, &deregisterFileSystem));

  services.insert(std::make_pair(9, &freeMemoryCallback));

  server.handle(&interpreterServices);

	return 0;
}
