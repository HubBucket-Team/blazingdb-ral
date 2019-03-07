#ifndef NODECOLUMNS_H_
#define NODECOLUMNS_H_

#include <blazingdb/communication/Node.h>
#include <GDFColumn.cuh>
#include <utility>

namespace ral {
namespace distribution {

struct NodeColumns {
  NodeColumns(blazingdb::communication::Node rNode,
              std::vector<gdf_column_cpp> rColumns)
      : node{std::move(rNode)}, columns{std::move(rColumns)} {}

  blazingdb::communication::Node node;
  std::vector<gdf_column_cpp> columns;
};

}  // namespace distribution
}  // namespace ral

#endif /* NODECOLUMNS_H_ */
