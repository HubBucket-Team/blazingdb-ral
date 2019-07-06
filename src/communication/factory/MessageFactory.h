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
                                                             const Node& sender_node,
                                                             std::uint64_t total_row_size,
                                                             std::vector<gdf_column_cpp>&& samples);

    static std::shared_ptr<Message> createSampleToNodeMaster(const ContextToken& context_token,
                                                             const Node& sender_node,
                                                             std::uint64_t total_row_size,
                                                             const std::vector<gdf_column_cpp>& samples);

    static std::shared_ptr<Message> createSampleToNodeMaster(std::shared_ptr<ContextToken>&& context_token,
                                                             const Node& sender_node,
                                                             std::uint64_t total_row_size,
                                                             std::vector<gdf_column_cpp>&& samples);

    static std::shared_ptr<Message> createSampleToNodeMaster(std::shared_ptr<ContextToken>&& context_token,
                                                             const Node& sender_node,
                                                             std::uint64_t total_row_size,
                                                             const std::vector<gdf_column_cpp>& samples);


    static std::shared_ptr<Message> createColumnDataMessage(const ContextToken& context_token,
                                                            const Node& sender_node,
                                                            std::vector<gdf_column_cpp>&& columns);

    static std::shared_ptr<Message> createColumnDataMessage(const ContextToken& context_token,
                                                            const Node& sender_node,
                                                            const std::vector<gdf_column_cpp>& columns);

    static std::shared_ptr<Message> createColumnDataMessage(std::shared_ptr<ContextToken>&& context_token,
                                                            const Node& sender_node,
                                                            std::vector<gdf_column_cpp>&& columns);

    static std::shared_ptr<Message> createColumnDataMessage(std::shared_ptr<ContextToken>&& context_token,
                                                            const Node& sender_node,
                                                            const std::vector<gdf_column_cpp>& columns);

    static std::shared_ptr<Message> createPartitionPivotsMessage(const ContextToken& context_token,
                                                                const Node& sender_node,
                                                                std::vector<gdf_column_cpp>&& columns);

    static std::shared_ptr<Message> createPartitionPivotsMessage(const ContextToken& context_token,
                                                                const Node& sender_node,
                                                                const std::vector<gdf_column_cpp>& columns);

    static std::shared_ptr<Message> createPartitionPivotsMessage(std::shared_ptr<ContextToken>&& context_token,
                                                                const Node& sender_node,
                                                                std::vector<gdf_column_cpp>&& columns);

    static std::shared_ptr<Message> createPartitionPivotsMessage(std::shared_ptr<ContextToken>&& context_token,
                                                                const Node& sender_node,
                                                                const std::vector<gdf_column_cpp>& columns);


    static std::shared_ptr<Message> createNodeDataMessage(const Node& selfNode);
};

} // namespace messages
} // namespace communication
} // namespace ral
