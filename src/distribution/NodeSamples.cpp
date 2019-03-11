#include "distribution/NodeSamples.h"

namespace ral {
namespace distribution {

NodeSamples::NodeSamples(std::size_t total_row_size,
                         Node&& sender_node,
                         std::vector<gdf_column_cpp>&& columns)
: total_row_size_{total_row_size},
  sender_node_{std::move(sender_node)},
  columns_{std::move(columns)}
{ }

NodeSamples::NodeSamples(std::size_t total_row_size,
                         const Node& sender_node,
                         std::vector<gdf_column_cpp>&& columns)
: total_row_size_{total_row_size},
  sender_node_{sender_node},
  columns_{std::move(columns)}
{ }

const std::size_t NodeSamples::getTotalRowSize() const {
    return total_row_size_;
}

const Node& NodeSamples::getSenderNode() const {
    return sender_node_;
}

std::vector<gdf_column_cpp> NodeSamples::getColumns() {
    return std::move(columns_);
}

} // namespace distribution
} // namespace ral
