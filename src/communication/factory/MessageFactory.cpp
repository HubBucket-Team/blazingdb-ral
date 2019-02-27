#include "communication/factory/MessageFactory.h"
#include "GDFColumn.cuh"

namespace ral {
namespace communication {
namespace messages {

    std::shared_ptr<SampleToNodeMasterMessage> MessageFactory::CreateSampleToNodeMaster(const Node& node,
                                                                                        const std::vector<gdf_column_cpp>& samples) {
        return std::make_shared<SampleToNodeMasterMessage>(node, samples);
    };

} // namespace messages
} // namespace communication
} // namespace ral