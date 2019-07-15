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

class NodeColumns {
public:
    NodeColumns(const Node& node, std::vector<gdf_column_cpp>&& columns);

public:
    NodeColumns(NodeColumns&& node_columns);

    NodeColumns& operator=(NodeColumns&& node_columns) = delete;

public:
    NodeColumns(const NodeColumns&) = delete;

    NodeColumns& operator=(const NodeColumns&) = delete;

public:
    const Node& getNode() const;

    /**
     * This function can only be called once due to it moves the internal container of columns to the client.
     *
     * @return  move a vector of gdf_column_cpp.
     */
    std::vector<gdf_column_cpp> getColumns();

    const std::vector<gdf_column_cpp>& getColumnsRef() const;

private:
    const Node node_;
    std::vector<gdf_column_cpp> columns_;
};

} // namespace distribution
} // namespace ral

#endif //BLAZINGDB_RAL_DISTRIBUTION_NODECOLUMNS_H
