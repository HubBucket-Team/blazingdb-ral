#ifndef BLAZINGDB_RAL_DISTRIBUTION_NODESAMPLES_H
#define BLAZINGDB_RAL_DISTRIBUTION_NODESAMPLES_H

#include <blazingdb/communication/Node.h>
#include <GDFColumn.cuh>

namespace ral {
namespace distribution {

namespace {
using Node = blazingdb::communication::Node;
} // namespace

class NodeSamples {
public:
    NodeSamples(std::size_t total_row_size, Node&& node, std::vector<gdf_column_cpp>&& columns);

    NodeSamples(std::size_t total_row_size, const Node& node, std::vector<gdf_column_cpp>&& columns);

public:
    const std::size_t getTotalRowSize() const;

    const Node& getSenderNode() const;

    /**
     * This function can only be called once due to it moves the internal container of columns to the client.
     *
     * @return  move a vector of gdf_column_cpp.
     */
    std::vector<gdf_column_cpp> getColumns();

    std::vector<gdf_column_cpp>& getColumnsRef();

    void setColumns(std::vector<gdf_column_cpp>&& columns);

private:
    const std::size_t total_row_size_;
    const Node sender_node_;
    std::vector<gdf_column_cpp> columns_;
};

}  // namespace distribution
}  // namespace ral

#endif // BLAZINGDB_RAL_DISTRIBUTION_NODESAMPLES_H
