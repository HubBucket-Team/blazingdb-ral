#include "distribution/NodeColumns2.h"

namespace ral {
namespace distribution {

NodeColumns2::NodeColumns2(const Node& node, std::vector<gdf_column_cpp>&& columns)
: node_{node}, columns_{std::move(columns)}
{ }

NodeColumns2::NodeColumns2(NodeColumns2&& node_columns)
: node_{node_columns.node_}, columns_{std::move(node_columns.columns_)}
{ }

const Node& NodeColumns2::getNode() const {
    return node_;
}

std::vector<gdf_column_cpp>& NodeColumns2::getColumns() {
    return columns_;
}

} // namespace distribution
} // namespace ral
