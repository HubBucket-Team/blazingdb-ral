/*
 * GDFParser.cpp
 *
 *  Created on: Apr 30, 2019
 *      Author: felipe
 */

#include "GDFParser.h"
#include "ral-message.cuh"
#include <blazingdb/protocol/message/utils.h>
#include <blazingdb/protocol/message/interpreter/gdf_dto.h>
#include <blazingdb/protocol/message/io/file_system.h>

namespace ral {
namespace io {



gdf_parser::gdf_parser(blazingdb::message::io::FileSystemBlazingTableSchema * table_schema, uint64_t accessToken) : access_token(accessToken) {
	// TODO Auto-generated constructor stub
	this->table_schema = table_schema;

	// WSM TODO table_schema news to be newed up and copy in the properties
}

gdf_parser::~gdf_parser() {
	// TODO Auto-generated destructor stub
	for(int i = 0; i < handles.size(); i++){
		if (handles[i]){
			cudaError_t error = cudaIpcCloseMemHandle(handles[i]);
			if (error != cudaSuccess){
				std::cout<<"Closing handle failed with error: "<<error<<std::endl;
			}
		} else {
			std::cout<<"Closing handle failed because the handle was null"<<std::endl;
		}
	}
}


void gdf_parser::parse(std::shared_ptr<arrow::io::RandomAccessFile> file,
		std::vector<gdf_column_cpp> & columns,
		const Schema & schema,
		std::vector<size_t> column_indices,
		size_t file_index){


	for(auto column : this->table_schema->gdf.columns) {

		gdf_column_cpp col;
		size_t column_index = 0;

		if (this->table_schema->gdf.columnTokens[column_index] == 0){
			//const std::string column_name = table.columnNames.at(column_index);
			const std::string column_name = "";

			if( ((gdf_dtype) column.dtype) == GDF_STRING){

		          nvstrings_ipc_transfer ipc;  // NOTE: IPC handles will be closed when nvstrings_ipc_transfer goes out of scope
		          memcpy(&ipc,column.custrings_data.data(),sizeof(nvstrings_ipc_transfer));

				NVStrings* strs = NVStrings::create_from_ipc(ipc);
				NVCategory* category = NVCategory::create_from_strings(*strs);
				NVStrings::destroy(strs);

				col.create_gdf_column(category, column.size, column_name);
			}
			else {
				// col.create_gdf_column_for_ipc((::gdf_dtype)column.dtype,libgdf::CudaIpcMemHandlerFrom(column.data),(gdf_valid_type*)libgdf::CudaIpcMemHandlerFrom(column.valid),column.size,column_name);
				col.create_gdf_column_for_ipc((gdf_dtype)column.dtype,libgdf::CudaIpcMemHandlerFrom(column.data),(gdf_valid_type *) libgdf::CudaIpcMemHandlerFrom(column.valid),column.size,column.null_count,column_name);
				handles.push_back(col.data());
			}

			if(col.valid() == nullptr){
				//TODO: we can remove this when libgdf properly
				//implements all algorithsm with valid == nullptr support
				//it crashes somethings like group by
				col.allocate_set_valid();

			}else{
				handles.push_back(col.valid());
			}
		}else{
			col = result_set_repository::get_instance().get_column(this->access_token, this->table_schema->gdf.columnTokens[column_index]);
		}

		columns.push_back(col);

		++column_index;
	}

	//call to blazing frame here
}

void gdf_parser::parse_schema(std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > files,
		ral::io::Schema & schema){
	std::vector<std::string> names;
	std::vector<gdf_dtype> types;

	std::for_each(this->table_schema->gdf.columns.begin(),this->table_schema->gdf.columns.end(),
			[&types](gdf_dto::gdf_column column)
			{
		types.push_back((gdf_dtype) column.dtype);
			});

	names = this->table_schema->tableSchema.names;
	ral::io::Schema temp_schema(names,types);
	schema = temp_schema;
	//generate schema from message here
}


}
}
