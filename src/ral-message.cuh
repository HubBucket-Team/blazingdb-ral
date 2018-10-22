#pragma once

#include <cuda_runtime.h>
#include <blazingdb/protocol/message/messages.h>
#include <blazingdb/protocol/message/interpreter/messages.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

namespace libgdf {

static std::basic_string<int8_t> BuildCudaIpcMemHandler (void *data) {
  cudaIpcMemHandle_t ipc_memhandle;
  cudaIpcGetMemHandle( &ipc_memhandle, (void*)data );
  cudaCheckErrors("Build IPC handle fail");

  std::basic_string<int8_t> bytes;
  bytes.resize(sizeof(cudaIpcMemHandle_t));
  memcpy((void*)bytes.data(), (int8_t*)(&ipc_memhandle), sizeof(cudaIpcMemHandle_t));
  return bytes;
}


static void* CudaIpcMemHandlerFrom (const std::basic_string<int8_t>& handler) {
  void * response = nullptr;
  std::cout << "handler-content: " <<  handler.size() <<  std::endl;
  if (handler.size() == 64) {
    cudaIpcMemHandle_t ipc_memhandle;
    memcpy((int8_t*)&ipc_memhandle, handler.data(), sizeof(ipc_memhandle));
    cudaIpcOpenMemHandle((void **)&response, ipc_memhandle, cudaIpcMemLazyEnablePeerAccess);
    cudaCheckErrors("From IPC handle fail");       
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
    for(auto column : table.columns) {

    	gdf_column_cpp col = gdf_column_cpp(libgdf::CudaIpcMemHandlerFrom(column.data), (gdf_valid_type*)libgdf::CudaIpcMemHandlerFrom(column.valid), (::gdf_dtype)column.dtype, (size_t)column.size, (gdf_size_type)column.null_count);
    	    	handles.push_back( col.data());
    	    	handles.push_back((void *) col.valid());
    	    	input_table.push_back(col);
    }
    input_tables.push_back(input_table);
  }

  return std::make_tuple(input_tables, table_names, column_names);
}

} //namespace libgdf
