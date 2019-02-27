#pragma once

#include "blazingdb/communication/messages/SampleToNodeMasterMessage.h"
#include "blazingdb/communication/messages/DataScatterMessage.h"
#include "blazingdb/communication/messages/PartitionPivotsMessage.h"
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

    using DataScatterMessage = Messages::DataScatterMessage<gdf_column_cpp,
                                                            gdf_column,
                                                            ral::communication::adapter::GpuFunctionsAdapter>;

    using DataPivot = Messages::DataPivot;
    using PartitionPivotsMessage = Messages::PartitionPivotsMessage;

} // namespace messages
} // namespace communication
} // namespace ral
