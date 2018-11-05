#pragma once

#include <cuda_runtime.h>

#include <blazingdb/protocol/message/messages.h>
#include <blazingdb/protocol/message/interpreter/messages.h>
#include "Utils.cuh"

namespace libgdf {

static std::basic_string<int8_t> BuildCudaIpcMemHandler (void *data) {
  std::basic_string<int8_t> bytes;
  if (data != nullptr) {
    cudaIpcMemHandle_t ipc_memhandle;
    CheckCudaErrors(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &ipc_memhandle, (void *) data));

    bytes.resize(sizeof(cudaIpcMemHandle_t));
    memcpy((void*)bytes.data(), (int8_t*)(&ipc_memhandle), sizeof(cudaIpcMemHandle_t));
  
  }
  return bytes;
}


static void* CudaIpcMemHandlerFrom (const std::basic_string<int8_t>& handler) {
  void * response = nullptr;
  std::cout << "handler-content: " <<  handler.size() <<  std::endl;
  if (handler.size() == 64) {
    cudaIpcMemHandle_t ipc_memhandle;
    memcpy((int8_t*)&ipc_memhandle, handler.data(), sizeof(ipc_memhandle));
    CheckCudaErrors(cudaIpcOpenMemHandle((void **)&response, ipc_memhandle, cudaIpcMemLazyEnablePeerAccess));
  }
  return response;
}

std::tuple<std::vector<std::vector<gdf_column_cpp>>,
           std::vector<std::string>,
           std::vector<std::vector<std::string>>> toBlazingDataframe(const ::blazingdb::protocol::TableGroupDTO& request,std::vector<void *> & handles)
{
  std::vector<std::vector<gdf_column_cpp>> input_tables;
  std::vector<std::string> table_names;
  std::vector<std::vector<std::string>> column_names;

  for(auto table : request.tables) {
    table_names.push_back(table.name);
    column_names.push_back(table.columnNames);

    std::vector<gdf_column_cpp> input_table;
    int column_index = 0;
    for(auto column : table.columns) {
    	const std::string column_name = table.columnNames.at(column_index);
//    	gdf_column_cpp col = gdf_column_cpp(libgdf::CudaIpcMemHandlerFrom(column.data), (gdf_valid_type*)libgdf::CudaIpcMemHandlerFrom(column.valid), (::gdf_dtype)column.dtype, (size_t)column.size, (gdf_size_type)column.null_count, column_name);
    	//    	handles.push_back((void *) col.valid());
    	//lines above commented becuase valid is set but its junk
    	gdf_column_cpp col;

    	col.create_gdf_column_for_ipc((::gdf_dtype)column.dtype,libgdf::CudaIpcMemHandlerFrom(column.data),nullptr,column.size,column_name);
    //	gdf_column_cpp col = gdf_column_cpp(libgdf::CudaIpcMemHandlerFrom(column.data), nullptr, (::gdf_dtype)column.dtype, (size_t)column.size, (gdf_size_type)column.null_count, column_name);
    	handles.push_back(col.data());

    	if(col.valid() == nullptr){
    		std::cout<<"oh thank god"<<std::endl;
    		 col.allocate_set_valid();
    	}
    	input_table.push_back(col);

    	++column_index;
    }
    input_tables.push_back(input_table);
  }

  return std::make_tuple(input_tables, table_names, column_names);
}

} //namespace libgdf
