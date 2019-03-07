#ifndef BLAZINGDB_RAL_DISTRIBUTION_NODECOLUMNS_H
#define BLAZINGDB_RAL_DISTRIBUTION_NODECOLUMNS_H

#include <vector>
#include "GDFColumn.cuh"
#include "blazingdb/communication/Node.h"

namespace ral {
namespace distribution {

namespace {
using Node = blazingdb::communication::Node;
} // namespace

class NodeColumns2 {
public:
    NodeColumns2(const Node& node, std::vector<gdf_column_cpp>&& columns);

public:
    NodeColumns2(NodeColumns2&& node_columns);

    NodeColumns2& operator=(NodeColumns2&& node_columns) = delete;

public:
    NodeColumns2(const NodeColumns2&) = delete;

    NodeColumns2& operator=(const NodeColumns2&) = delete;

public:
    const Node& getNode() const;

    std::vector<gdf_column_cpp>& getColumns();

private:
    const Node node_;
    std::vector<gdf_column_cpp> columns_;
};

} // namespace distribution
} // namespace ral

#endif //BLAZINGDB_RAL_DISTRIBUTION_NODECOLUMNS_H
