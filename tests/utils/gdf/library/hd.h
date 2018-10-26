#ifndef RAL_TESTS_UTILS_GDF_HD_H_
#define RAL_TESTS_UTILS_GDF_HD_H_

#include <vector>

#include <gdf/gdf.h>

namespace gdf {
namespace library {

template <gdf_dtype U>
std::vector<typename DType<U>::value_type>
HostVectorFrom(gdf_column_cpp &column) {
  std::vector<typename DType<U>::value_type> vector;
  vector.reserve(column.size());
  cudaMemcpy(vector.data(),
             column.data(),
             column.size() * DType<U>::size,
             cudaMemcpyDeviceToHost);
  return vector;
}

}  // namespace library
}  // namespace gdf

#endif
