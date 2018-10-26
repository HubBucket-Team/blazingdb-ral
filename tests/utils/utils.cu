#include "utils.h"

namespace ral {
namespace test {
namespace utils {

Column::~Column() {}

gdf_column_cpp Column::Create(const gdf_dtype   dtype,
                              const std::size_t length,
                              const void *      data,
                              const std::size_t size) {
  gdf_column_cpp column_cpp;
  column_cpp.create_gdf_column(dtype, length, const_cast<void *>(data), size);
  return column_cpp;
}

std::vector<gdf_column_cpp> Table::ToGdfColumnCpps() const {
  std::vector<gdf_column_cpp> gdfColumnsCpps;
  gdfColumnsCpps.resize(columns_.size());
  std::transform(columns_.cbegin(),
                 columns_.cend(),
                 gdfColumnsCpps.begin(),
                 [](const std::shared_ptr<Column> &column) {
                   return column->ToGdfColumnCpp();
                 });
  return gdfColumnsCpps;
}

BlazingFrame TableGroup::ToBlazingFrame() const {
  BlazingFrame frame;
  frame.resize(tables_.size());
  std::transform(tables_.cbegin(),
                 tables_.cend(),
                 frame.begin(),
                 [](const Table &table) { return table.ToGdfColumnCpps(); });
  return frame;
}

}  // namespace utils
}  // namespace test
}  // namespace ral
