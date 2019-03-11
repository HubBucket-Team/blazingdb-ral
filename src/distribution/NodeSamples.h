#ifndef NODESAMPLES_H_
#define NODESAMPLES_H_

#include <blazingdb/communication/Node.h>
#include <GDFColumn.cuh>

namespace ral {
namespace distribution {

struct NodeSamples {
  NodeSamples(blazingdb::communication::Node rNode,
              std::vector<gdf_column_cpp> rColumns, int rowSize)
      : node{std::move(rNode)},
        columns{std::move(rColumns)},
        row_size{rowSize} {}

  blazingdb::communication::Node node;
  int row_size;
  std::vector<gdf_column_cpp> columns;
};

}  // namespace distribution
}  // namespace ral

#endif /* NODECOLUMNS_H_ */
