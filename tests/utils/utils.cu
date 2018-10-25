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

BlazingFrame TableGroup::ToBlazingFrame() const {
  BlazingFrame frame;
  return frame;
}

}  // namespace utils
}  // namespace test
}  // namespace ral
