#pragma once

#include "blazingdb/communication/messages/SampleToNodeMasterMessage.h"
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
} // namespace messages
} // namespace communication
} // namespace ral
