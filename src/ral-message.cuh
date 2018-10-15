#pragma once

#include <cuda_runtime.h>
#include <blazingdb/protocol/messages.h>

using namespace libgdf;

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

namespace blazingdb {
namespace protocol {

static flatbuffers::Offset<flatbuffers::Vector<int8_t>> _BuildCudaIpcMemHandler (flatbuffers::FlatBufferBuilder &builder, void *data) {
  flatbuffers::Offset<flatbuffers::Vector<int8_t>> offsets;
  cudaIpcMemHandle_t ipc_memhandle;
  cudaIpcGetMemHandle( &ipc_memhandle, (void*)data );
  cudaCheckErrors("IPC handle fail");

  int8_t* bytes = new int8_t[sizeof(cudaIpcMemHandle_t)];
  memcpy(bytes, (int8_t*)(&ipc_memhandle), sizeof(cudaIpcMemHandle_t));

  return builder.CreateVector(bytes, sizeof(cudaIpcMemHandle_t));
}

static void* _CudaIpcMemHandlerFrom (const gdf::cudaIpcMemHandle_t *handler) {
  void * response = nullptr;
  cudaIpcMemHandle_t ipc_memhandle;

  auto bytes = handler->reserved();
  memcpy((int8_t*)&ipc_memhandle, bytes->data(), sizeof(ipc_memhandle));
  cudaIpcOpenMemHandle((void **)&response, ipc_memhandle, cudaIpcMemLazyEnablePeerAccess);
  cudaCheckErrors("IPC handle fail");

  return response;
}


static TableGroupDTO _TableGroupDTOFrom(const blazingdb::protocol::TableGroup * tableGroup) {
  std::string name = std::string{tableGroup->name()->c_str()};
  std::vector<BlazingTableDTO> tables;

  auto rawTables = tableGroup->tables();
  for (const auto& table : *rawTables) {
    std::vector<::libgdf::gdf_column>  columns;
    std::vector<std::string>  columnNames;

    for (const auto& c : *table->columns()){
      ::libgdf::gdf_column column = {
          .data = _CudaIpcMemHandlerFrom(c->data()),
          .valid = (unsigned char *)_CudaIpcMemHandlerFrom(c->valid()),
          .size = c->size(),
          .dtype = (libgdf::gdf_dtype)c->dtype(),
          .null_count = c->null_count(),
          .dtype_info = libgdf::gdf_dtype_extra_info {
             .time_unit = (libgdf::gdf_time_unit) c->dtype_info()->time_unit()
          },
      };
      columns.push_back(column);
    }

    tables.push_back(BlazingTableDTO{
        .name = std::string{table->name()->c_str()},
        .columns = columns,
        .columnNames = columnNames,
    });
  }

  return TableGroupDTO {
    .tables = tables,
    .name = name,
  };
}

static flatbuffers::Offset<TableGroup> _BuildTableGroup(flatbuffers::FlatBufferBuilder &builder,
                                                       const TableGroupDTO &tableGroup) {
  auto tableNameOffset = builder.CreateString(tableGroup.name);
  std::vector<flatbuffers::Offset<BlazingTable>> tablesOffset;

  auto _createColumns = [] (flatbuffers::FlatBufferBuilder &builder, std::vector<::libgdf::gdf_column> &columns) -> std::vector<flatbuffers::Offset<gdf::gdf_column_handler>> {
    std::vector<flatbuffers::Offset<gdf::gdf_column_handler>> offsets;
    for (auto & c: columns) {
      auto dtype_extra_info = gdf::Creategdf_dtype_extra_info (builder, (gdf::gdf_time_unit)c.dtype_info.time_unit );
      auto data_offset = gdf::CreatecudaIpcMemHandle_t(builder, _BuildCudaIpcMemHandler (builder, c.data) );
      auto valid_offset = gdf::CreatecudaIpcMemHandle_t(builder, _BuildCudaIpcMemHandler(builder, c.valid) );
      auto column_offset = ::blazingdb::protocol::gdf::Creategdf_column_handler(builder, data_offset, valid_offset, c.size, (gdf::gdf_dtype)c.dtype, dtype_extra_info);
      offsets.push_back(column_offset);
    }
    return offsets;
  };
  auto _createColumnNames  = [] (flatbuffers::FlatBufferBuilder &builder, std::vector<std::string> &columnNames) -> std::vector<flatbuffers::Offset<flatbuffers::String>> {
    std::vector<flatbuffers::Offset<flatbuffers::String>> offsets;
    for (auto & name: columnNames) {
      offsets.push_back( builder.CreateString(name.data()));
    }
    return offsets;
  };
  for (auto table : tableGroup.tables) {
    auto columns = _createColumns(builder, table.columns);
    auto columnNames = _createColumnNames(builder, table.columnNames);
    tablesOffset.push_back( CreateBlazingTable(builder, builder.CreateString(table.name), builder.CreateVector(columns), builder.CreateVector(columnNames)));
  }

  auto tables = builder.CreateVector(tablesOffset);
  return CreateTableGroup(builder, tables, tableNameOffset);
}
 

class RalRequestMessage  : public IMessage {
public:

  RalRequestMessage(const std::string &logicalPlan, const  ::blazingdb::protocol::TableGroupDTO &tableGroup)
      : IMessage(), logicalPlan{logicalPlan}, tableGroup{tableGroup}
  {

  }
  RalRequestMessage (const uint8_t* buffer)
      : IMessage()
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::interpreter::DMLRequest>(buffer);
    logicalPlan = std::string{pointer->logicalPlan()->c_str()};
    tableGroup =  _TableGroupDTOFrom(pointer->tableGroup());
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder;
    auto logicalPlan_offset = builder.CreateString(logicalPlan);
    auto tableGroupOffset = _BuildTableGroup(builder, tableGroup);
    builder.Finish(interpreter::CreateDMLRequest(builder, logicalPlan_offset, tableGroupOffset));
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  std::string getLogicalPlan() {
    return logicalPlan;
  }

  ::blazingdb::protocol::TableGroupDTO getTableGroup() {
    return tableGroup;
  }

private:
  std::string logicalPlan;
  ::blazingdb::protocol::TableGroupDTO tableGroup;
};

}
}
