#include "communication/factory/MessageFactory.h"
#include "GDFColumn.cuh"

namespace ral {
namespace communication {
namespace messages {

    std::shared_ptr<Message> Factory::createSampleToNodeMaster(const ContextToken& context_token,
                                                               const Node& node,
                                                               std::uint64_t total_data_size,
                                                               std::vector<gdf_column_cpp>&& samples) {
        return std::make_shared<SampleToNodeMasterMessage>(context_token, node, total_data_size, std::move(samples));
    }

    std::shared_ptr<Message> Factory::createSampleToNodeMaster(const ContextToken& context_token,
                                                               const Node& node,
                                                               std::uint64_t total_data_size,
                                                               const std::vector<gdf_column_cpp>& samples) {
        return std::make_shared<SampleToNodeMasterMessage>(context_token, node, total_data_size, samples);
    };

    std::shared_ptr<Message> Factory::createSampleToNodeMaster(std::unique_ptr<ContextToken>&& context_token,
                                                               const Node& node,
                                                               std::uint64_t total_data_size,
                                                               std::vector<gdf_column_cpp>&& samples) {
        return std::make_shared<SampleToNodeMasterMessage>(std::move(context_token), node, total_data_size, std::move(samples));
    }

    std::shared_ptr<Message> Factory::createSampleToNodeMaster(std::unique_ptr<ContextToken>&& context_token,
                                                               const Node& node,
                                                               std::uint64_t total_data_size,
                                                               const std::vector<gdf_column_cpp>& samples) {
        return std::make_shared<SampleToNodeMasterMessage>(std::move(context_token), node, total_data_size, samples);
    };

    std::shared_ptr<Message> Factory::createDataScatterMessage(const ContextToken& context_token,
                                                               std::vector<gdf_column_cpp>&& columns) {
        return std::make_shared<DataScatterMessage>(context_token, std::move(columns));
    }

    std::shared_ptr<Message> Factory::createDataScatterMessage(const ContextToken& context_token,
                                                               const std::vector<gdf_column_cpp>& columns) {
        return std::make_shared<DataScatterMessage>(context_token, columns);
    }

    std::shared_ptr<Message> Factory::createDataScatterMessage(std::unique_ptr<ContextToken>&& context_token,
                                                               std::vector<gdf_column_cpp>&& columns) {
        return std::make_shared<DataScatterMessage>(std::move(context_token), std::move(columns));
    }

    std::shared_ptr<Message> Factory::createDataScatterMessage(std::unique_ptr<ContextToken>&& context_token,
                                                               const std::vector<gdf_column_cpp>& columns) {
        return std::make_shared<DataScatterMessage>(std::move(context_token), columns);
    }

    DataPivot Factory::createDataPivot(const Node& node, std::string&& min_range, std::string&& max_range) {
        return DataPivot(node, std::move(min_range), std::move(max_range));
    }

    DataPivot Factory::createDataPivot(const Node& node, const std::string& min_range, const std::string& max_range) {
        return DataPivot(node, min_range, max_range);
    }

    std::shared_ptr<Message> Factory::createPartitionPivotsMessage(const ContextToken& context_token,
                                                                   std::vector<DataPivot>&& pivots) {
        return std::make_shared<PartitionPivotsMessage>(context_token, std::move(pivots));
    }

    std::shared_ptr<Message> Factory::createPartitionPivotsMessage(const ContextToken& context_token,
                                                                   const std::vector<DataPivot>& pivots) {
        return std::make_shared<PartitionPivotsMessage>(context_token, pivots);
    }

    std::shared_ptr<Message> Factory::createPartitionPivotsMessage(std::unique_ptr<ContextToken>&& context_token,
                                                                   std::vector<DataPivot>&& pivots) {
        return std::make_shared<PartitionPivotsMessage>(std::move(context_token), std::move(pivots));
    }

    std::shared_ptr<Message> Factory::createPartitionPivotsMessage(std::unique_ptr<ContextToken>&& context_token,
                                                                   const std::vector<DataPivot>& pivots) {
        return std::make_shared<PartitionPivotsMessage>(std::move(context_token), pivots);
    }

    std::shared_ptr<Message> Factory::createNodeDataMessage(const Node& selfNode) {
        return std::make_shared<NodeDataMessage>(selfNode);
    }

} // namespace messages
} // namespace communication
} // namespace ral
