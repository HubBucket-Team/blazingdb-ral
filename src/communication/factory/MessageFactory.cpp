#include "communication/factory/MessageFactory.h"
#include "GDFColumn.cuh"

namespace ral {
namespace communication {
namespace messages {

    std::shared_ptr<Message> Factory::createSampleToNodeMaster(const Node& node,
                                                               std::vector<gdf_column_cpp>&& samples) {
        return std::make_shared<SampleToNodeMasterMessage>(node, std::move(samples));
    }

    std::shared_ptr<Message> Factory::createSampleToNodeMaster(const Node& node,
                                                               const std::vector<gdf_column_cpp>& samples) {
        return std::make_shared<SampleToNodeMasterMessage>(node, samples);
    };

    std::shared_ptr<Message> Factory::createDataScatterMessage(std::vector<gdf_column_cpp>&& columns) {
        return std::make_shared<DataScatterMessage>(std::move(columns));
    }

    std::shared_ptr<Message> Factory::createDataScatterMessage(const std::vector<gdf_column_cpp>& columns) {
        return std::make_shared<DataScatterMessage>(columns);
    }

    DataPivot Factory::createDataPivot(const Node& node, std::string&& min_range, std::string&& max_range) {
        return DataPivot(node, std::move(min_range), std::move(max_range));
    }

    DataPivot Factory::createDataPivot(const Node& node, const std::string& min_range, const std::string& max_range) {
        return DataPivot(node, min_range, max_range);
    }

    std::shared_ptr<Message> Factory::createPartitionPivotsMessage(std::vector<DataPivot>&& pivots) {
        return std::make_shared<PartitionPivotsMessage>(std::move(pivots));
    }

    std::shared_ptr<Message> Factory::createPartitionPivotsMessage(const std::vector<DataPivot>& pivots) {
        return std::make_shared<PartitionPivotsMessage>(pivots);
    }

    std::shared_ptr<Message> Factory::createNodeDataMessage(const Node& selfNode) {
        return std::make_shared<NodeDataMessage>(selfNode);
    }

} // namespace messages
} // namespace communication
} // namespace ral
