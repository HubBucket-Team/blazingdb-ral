#include "distribution/NodeColumns.h"

namespace ral {
namespace distribution {

NodeColumns::NodeColumns(const Node& node, std::vector<gdf_column_cpp>&& columns)
: node_{node}, columns_{std::move(columns)}
{ }

NodeColumns::NodeColumns(NodeColumns&& node_columns)
: node_{node_columns.node_}, columns_{std::move(node_columns.columns_)}
{ }

const Node& NodeColumns::getNode() const {
    return node_;
}

std::vector<gdf_column_cpp> NodeColumns::getColumns() {
    return std::move(columns_);
}

} // namespace distribution
} // namespace ral
