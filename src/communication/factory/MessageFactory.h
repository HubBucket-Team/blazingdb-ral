#pragma once

#include <vector>
#include "communication/messages/ComponentMessages.h"
#include "blazingdb/communication/Node.h"

namespace ral {
namespace communication {
namespace messages {

    namespace {
        using Node = blazingdb::communication::Node;
        using Message = blazingdb::communication::messages::Message;
    }

    struct Factory {
        static std::shared_ptr<Message> createSampleToNodeMaster(const Node& node,
                                                                 std::vector<gdf_column_cpp>&& samples);

        static std::shared_ptr<Message> createSampleToNodeMaster(const Node& node,
                                                                 const std::vector<gdf_column_cpp>& samples);


        static std::shared_ptr<Message> createDataScatterMessage(std::vector<gdf_column_cpp>&& columns);

        static std::shared_ptr<Message> createDataScatterMessage(const std::vector<gdf_column_cpp>& columns);


        static DataPivot createDataPivot(const Node& node, std::string&& min_range, std::string&& max_range);

        static DataPivot createDataPivot(const Node& node, const std::string& min_range, const std::string& max_range);

        static std::shared_ptr<Message> createPartitionPivotsMessage(std::vector<DataPivot>&& pivots);

        static std::shared_ptr<Message> createPartitionPivotsMessage(const std::vector<DataPivot>& pivots);

        template <typename Type>
        static std::shared_ptr<Type> make(const std::string& json_data, const std::string& binary_data) {
            return Type::make(json_data, binary_data);
        }
    };

} // namespace messages
} // namespace communication
} // namespace ral
