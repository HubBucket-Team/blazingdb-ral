#pragma once

#include <vector>
#include "communication/messages/ComponentMessages.h"
#include "blazingdb/communication/Node.h"

namespace ral {
namespace communication {
namespace messages {

    using Node = blazingdb::communication::Node;

    struct Factory {
        static std::shared_ptr<SampleToNodeMasterMessage> createSampleToNodeMaster(const Node& node,
                                                                                   std::vector<gdf_column_cpp>&& samples);

        static std::shared_ptr<SampleToNodeMasterMessage> createSampleToNodeMaster(const Node& node,
                                                                                   const std::vector<gdf_column_cpp>& samples);


        static std::shared_ptr<DataScatterMessage> createDataScatterMessage(std::vector<gdf_column_cpp>&& columns);

        static std::shared_ptr<DataScatterMessage> createDataScatterMessage(const std::vector<gdf_column_cpp>& columns);


        static DataPivot createDataPivot(const Node& node, std::string&& min_range, std::string&& max_range);

        static DataPivot createDataPivot(const Node& node, const std::string& min_range, const std::string& max_range);

        static std::shared_ptr<PartitionPivotsMessage> createPartitionPivotsMessage(std::vector<DataPivot>&& pivots);

        static std::shared_ptr<PartitionPivotsMessage> createPartitionPivotsMessage(const std::vector<DataPivot>& pivots);
    };

} // namespace messages
} // namespace communication
} // namespace ral
