#pragma once

#include "blazingdb/communication/messages/ColumnDataMessage.h"
#include "blazingdb/communication/messages/SampleToNodeMasterMessage.h"
#include "blazingdb/communication/messages/DataScatterMessage.h"
#include "blazingdb/communication/messages/NodeDataMessage.h"
#include "blazingdb/communication/messages/PartitionPivotsMessage.h"
#include "blazingdb/communication/messages/DataPivot.h"
#include "communication/adapter/GpuFunctionsAdapter.h"

namespace ral {
namespace communication {
namespace messages {

    namespace {
        namespace Messages = blazingdb::communication::messages;
    }

    using SampleToNodeMasterMessage = Messages::SampleToNodeMasterMessage<gdf_column_cpp,
                                                                          gdf_column,
                                                                          ral::communication::adapter::GpuFunctionsAdapter>;

    using ColumnDataMessage = Messages::ColumnDataMessage<gdf_column_cpp,
                                                          gdf_column,
                                                          ral::communication::adapter::GpuFunctionsAdapter>;
    
    using PartitionPivotsMessage = Messages::PartitionPivotsMessage<gdf_column_cpp,
                                                                gdf_column,
                                                                ral::communication::adapter::GpuFunctionsAdapter>;

    using DataScatterMessage = Messages::DataScatterMessage<gdf_column_cpp,
                                                            gdf_column,
                                                            ral::communication::adapter::GpuFunctionsAdapter>;

    using DataPivot = Messages::DataPivot;

    using NodeDataMessage = Messages::NodeDataMessage;

} // namespace messages
} // namespace communication
} // namespace ral
