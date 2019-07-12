#include <algorithm>

#include "communication/adapter/GpuFunctionsAdapter.h"
#include "Traits/RuntimeTraits.h"
#include "GDFColumn.cuh"

namespace ral {
namespace communication {
namespace adapter {

    void GpuFunctionsAdapter::copyGpuToCpu(std::size_t& binary_pointer, std::string& result, gdf_column_cpp& column)
    {
        if (column.size() == 0) {
            return;
        }

        if (isGdfString(*column.get_gdf_column())) {
          NVCategory * nvCategory = reinterpret_cast<NVCategory *>(
              column.get_gdf_column()->dtype_info.category);

          NVStrings * nvStrings = nvCategory->to_strings();

          const std::size_t stringsLength = nvStrings->size();
          const std::size_t offsetsLength = stringsLength + 1;

          int * const lengthPerStrings = new int[stringsLength];
          nvStrings->byte_count(lengthPerStrings, false);

          const std::size_t stringsSize = std::accumulate(
              lengthPerStrings, lengthPerStrings + stringsLength, 0);
          const std::size_t offsetsSize = offsetsLength * sizeof(int);

          char * stringsPointer = new char[stringsSize];
          int *  offsetsPointer = new int[offsetsSize];

          nvStrings->create_offsets(
              stringsPointer, offsetsPointer, nullptr, false);

          const std::size_t totalSize =
              stringsSize + offsetsSize + 3 * sizeof(const std::size_t);

          binary_pointer += totalSize;
          result.reserve(totalSize);
          std::memcpy(&result[0], &stringsSize, sizeof(const std::size_t));
          std::memcpy(&result[sizeof(const std::size_t)],
                      &offsetsSize,
                      sizeof(const std::size_t));
          std::memcpy(&result[2 * sizeof(const std::size_t)],
                      &stringsLength,
                      sizeof(const std::size_t));
          std::memcpy(&result[3 * sizeof(const std::size_t)],
                      stringsPointer,
                      stringsSize);
          std::memcpy(&result[3 * sizeof(const std::size_t) + stringsSize],
                      offsetsPointer,
                      offsetsSize);

          // TODO: remove pointers to map into `result` without bypass
          delete stringsPointer;
          delete[] offsetsPointer;
          NVStrings::destroy(nvStrings);
        } else {
          std::size_t data_size = getDataCapacity(column.get_gdf_column());
          cudaMemcpy(&result[binary_pointer],
                     column.data(),
                     data_size,
                     cudaMemcpyDeviceToHost);
          binary_pointer += data_size;

          std::size_t valid_size = getValidCapacity(column.get_gdf_column());
          cudaMemcpy(&result[binary_pointer],
                     column.valid(),
                     valid_size,
                     cudaMemcpyDeviceToHost);
          binary_pointer += valid_size;
        }
    }

    std::size_t GpuFunctionsAdapter::getDataCapacity(gdf_column* column) {
        return ral::traits::get_data_size_in_bytes(column);
    }

    std::size_t GpuFunctionsAdapter::getValidCapacity(gdf_column* column) {
        return ral::traits::get_bitmask_size_in_bytes(column->size);
    }

    std::size_t GpuFunctionsAdapter::getDTypeSize(gdf_dtype dtype) {
        return ral::traits::get_dtype_size_in_bytes(dtype);
    }

    std::size_t GpuFunctionsAdapter::isGdfString(const gdf_column & column) {
      return (GDF_STRING == column.dtype) ||
             (GDF_STRING_CATEGORY == column.dtype);
    }

    NVStrings *
    GpuFunctionsAdapter::CreateNvStrings(const void *      stringsPointer,
                                         const void *      offsetsPointer,
                                         const std::size_t keysLength) {
      return NVStrings::create_from_offsets(
          reinterpret_cast<const char *>(stringsPointer),
          keysLength,
          reinterpret_cast<const int *>(offsetsPointer),
          nullptr,
          0);
    }

} // namespace adapter
} // namespace communication
} // namespace ral
