#pragma once

#include <vector>
#include "communication/messages/ComponentMessages.h"
#include "blazingdb/communication/ContextToken.h"
#include "blazingdb/communication/Node.h"

namespace ral {
namespace communication {
namespace messages {

    namespace {
        using Node = blazingdb::communication::Node;
        using Message = blazingdb::communication::messages::Message;
        using ContextToken = blazingdb::communication::ContextToken;
    }

    struct Factory {
        static std::shared_ptr<Message> createSampleToNodeMaster(const ContextToken& context_token,
                                                                 const Node& node,
                                                                 std::uint64_t total_data_size,
                                                                 std::vector<gdf_column_cpp>&& samples);

        static std::shared_ptr<Message> createSampleToNodeMaster(const ContextToken& context_token,
                                                                 const Node& node,
                                                                 std::uint64_t total_data_size,
                                                                 const std::vector<gdf_column_cpp>& samples);

        static std::shared_ptr<Message> createSampleToNodeMaster(std::unique_ptr<ContextToken>&& context_token,
                                                                 const Node& node,
                                                                 std::uint64_t total_data_size,
                                                                 std::vector<gdf_column_cpp>&& samples);

        static std::shared_ptr<Message> createSampleToNodeMaster(std::unique_ptr<ContextToken>&& context_token,
                                                                 const Node& node,
                                                                 std::uint64_t total_data_size,
                                                                 const std::vector<gdf_column_cpp>& samples);


        static std::shared_ptr<Message> createDataScatterMessage(const ContextToken& context_token,
                                                                 std::vector<gdf_column_cpp>&& columns);

        static std::shared_ptr<Message> createDataScatterMessage(const ContextToken& context_token,
                                                                 const std::vector<gdf_column_cpp>& columns);

        static std::shared_ptr<Message> createDataScatterMessage(std::unique_ptr<ContextToken>&& context_token,
                                                                 std::vector<gdf_column_cpp>&& columns);

        static std::shared_ptr<Message> createDataScatterMessage(std::unique_ptr<ContextToken>&& context_token,
                                                                 const std::vector<gdf_column_cpp>& columns);


        static DataPivot createDataPivot(const Node& node, std::string&& min_range, std::string&& max_range);

        static DataPivot createDataPivot(const Node& node, const std::string& min_range, const std::string& max_range);

        static std::shared_ptr<Message> createPartitionPivotsMessage(const ContextToken& context_token,
                                                                     std::vector<DataPivot>&& pivots);

        static std::shared_ptr<Message> createPartitionPivotsMessage(const ContextToken& context_token,
                                                                     const std::vector<DataPivot>& pivots);

        static std::shared_ptr<Message> createPartitionPivotsMessage(std::unique_ptr<ContextToken>&& context_toke,
                                                                     std::vector<DataPivot>&& pivots);

        static std::shared_ptr<Message> createPartitionPivotsMessage(std::unique_ptr<ContextToken>&& context_toke,
                                                                     const std::vector<DataPivot>& pivots);


        static std::shared_ptr<Message> createNodeDataMessage(const Node& selfNode);
    };

} // namespace messages
} // namespace communication
} // namespace ral
