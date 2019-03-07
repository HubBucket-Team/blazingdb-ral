#include "communication/factory/MessageFactory.h"
#include "GDFColumn.cuh"

namespace ral {
namespace communication {
namespace messages {

namespace {
using MessageToken = blazingdb::communication::messages::MessageToken;
} // namespace

std::shared_ptr<Message> Factory::createSampleToNodeMaster(const ContextToken& context_token,
                                                           const Node& sender_node,
                                                           std::uint64_t total_row_size,
                                                           std::vector<gdf_column_cpp>&& samples) {
    std::unique_ptr<MessageToken> message_token_pointer = MessageToken::Make(SampleToNodeMasterMessage::getMessageID());
    std::unique_ptr<ContextToken> context_token_pointer = ContextToken::Make(context_token.getIntToken());
    return std::make_shared<SampleToNodeMasterMessage>(std::move(message_token_pointer),
                                                       std::move(context_token_pointer),
                                                       sender_node,
                                                       total_row_size,
                                                       std::move(samples));
}

std::shared_ptr<Message> Factory::createSampleToNodeMaster(const ContextToken& context_token,
                                                           const Node& sender_node,
                                                           std::uint64_t total_row_size,
                                                           const std::vector<gdf_column_cpp>& samples) {
    std::unique_ptr<MessageToken> message_token_pointer = MessageToken::Make(SampleToNodeMasterMessage::getMessageID());
    std::unique_ptr<ContextToken> context_token_pointer = ContextToken::Make(context_token.getIntToken());
    return std::make_shared<SampleToNodeMasterMessage>(std::move(message_token_pointer),
                                                       std::move(context_token_pointer),
                                                       sender_node,
                                                       total_row_size,
                                                       samples);
};

std::shared_ptr<Message> Factory::createSampleToNodeMaster(std::unique_ptr<ContextToken>&& context_token,
                                                           const Node& sender_node,
                                                           std::uint64_t total_row_size,
                                                           std::vector<gdf_column_cpp>&& samples) {
    std::unique_ptr<MessageToken> message_token = MessageToken::Make(SampleToNodeMasterMessage::getMessageID());
    return std::make_shared<SampleToNodeMasterMessage>(std::move(message_token),
                                                       std::move(context_token),
                                                       sender_node,
                                                       total_row_size,
                                                       std::move(samples));
}

std::shared_ptr<Message> Factory::createSampleToNodeMaster(std::unique_ptr<ContextToken>&& context_token,
                                                           const Node& sender_node,
                                                           std::uint64_t total_row_size,
                                                           const std::vector<gdf_column_cpp>& samples) {
    std::unique_ptr<MessageToken> message_token = MessageToken::Make(SampleToNodeMasterMessage::getMessageID());
    return std::make_shared<SampleToNodeMasterMessage>(std::move(message_token),
                                                       std::move(context_token),
                                                       sender_node,
                                                       total_row_size,
                                                       samples);
};


std::shared_ptr<Message> Factory::createColumnDataMessage(const ContextToken& context_token,
                                                          const Node& sender_node,
                                                          std::vector<gdf_column_cpp>&& columns) {
    std::unique_ptr<MessageToken> message_token_pointer = MessageToken::Make(ColumnDataMessage::getMessageID());
    std::unique_ptr<ContextToken> context_token_pointer = ContextToken::Make(context_token.getIntToken());
    return std::make_shared<ColumnDataMessage>(std::move(message_token_pointer),
                                               std::move(context_token_pointer),
                                               sender_node,
                                               std::move(columns));
}

std::shared_ptr<Message> Factory::createColumnDataMessage(const ContextToken& context_token,
                                                          const Node& sender_node,
                                                          const std::vector<gdf_column_cpp>& columns) {
    std::unique_ptr<MessageToken> message_token_pointer = MessageToken::Make(ColumnDataMessage::getMessageID());
    std::unique_ptr<ContextToken> context_token_pointer = ContextToken::Make(context_token.getIntToken());
    return std::make_shared<ColumnDataMessage>(std::move(message_token_pointer),
                                               std::move(context_token_pointer),
                                               sender_node,
                                               columns);
}

std::shared_ptr<Message> Factory::createColumnDataMessage(std::unique_ptr<ContextToken>&& context_token,
                                                          const Node& sender_node,
                                                          std::vector<gdf_column_cpp>&& columns) {
    std::unique_ptr<MessageToken> message_token = MessageToken::Make(ColumnDataMessage::getMessageID());
    return std::make_shared<ColumnDataMessage>(std::move(message_token),
                                               std::move(context_token),
                                               sender_node,
                                               std::move(columns));
}

std::shared_ptr<Message> Factory::createColumnDataMessage(std::unique_ptr<ContextToken>&& context_token,
                                                          const Node& sender_node,
                                                          const std::vector<gdf_column_cpp>& columns) {
    std::unique_ptr<MessageToken> message_token = MessageToken::Make(ColumnDataMessage::getMessageID());
    return std::make_shared<ColumnDataMessage>(std::move(message_token),
                                               std::move(context_token),
                                               sender_node,
                                               columns);
}


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
