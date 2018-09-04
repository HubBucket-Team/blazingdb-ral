/*
 * MessageBuilder.cpp
 *
 *  Created on: Jul 14, 2017
 *      Author: BlazingDB
 */

#include "Protocol/MessageBuilder.h"

#include "cudaIpcMemHandle_t_generated.h"
#include "gdf_column_generated.h"
#include "gdf_dtype_extra_info_generated.h"
#include "gdf_dtype_generated.h"
#include "gdf_time_unit_generated.h"

BlazingProtocol::Payload::Payload(uint8_t* data, uint32_t size)
    : m_data(data), m_size(size) {
}

const BlazingProtocol::Payload BlazingProtocol::buildServerMessage(const std::string &statement, const std::string &authorization, const QueryResult &queryResult) {





    // Build up a serialized buffer
    const int initial_buffer_size = 1024; // bytes
    flatbuffers::FlatBufferBuilder builder(initial_buffer_size);

    std::vector<flatbuffers::Offset<BlazingDB::Protocol::Column>> columns_vector;

    auto statementString = builder.CreateString(statement);
    auto success = true;
    auto message = builder.CreateString("cat is sleeping!");
    //auto resultset = BlazingDB::Protocol::CreateResultSet(builder, columns, rows);
    auto authorizationString = builder.CreateString(authorization);

    builder.Finish(message);

    const BlazingProtocol::Payload payload(builder.GetBufferPointer(), builder.GetSize());

    return payload;
}
