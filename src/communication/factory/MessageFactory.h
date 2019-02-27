#pragma once

#include <vector>
#include "communication/messages/ComponentMessages.h"
#include "blazingdb/communication/Node.h"

namespace ral {
namespace communication {
namespace messages {

    using Node = blazingdb::communication::Node;

    struct MessageFactory {
        std::shared_ptr<SampleToNodeMasterMessage> CreateSampleToNodeMaster(const Node& node,
                                                                            const std::vector<gdf_column_cpp>& samples);
    };

} // namespace messages
} // namespace communication
} // namespace ral
