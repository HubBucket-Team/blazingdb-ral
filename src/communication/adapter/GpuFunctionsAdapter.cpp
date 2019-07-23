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
        // std::cout<<"copyGpuToCpu column size"<<column.size()<<std::endl;
        // std::cout<<"copyGpuToCpu column dtype"<<column.dtype()<<std::endl;
        // std::cout<<"copyGpuToCpu column null_count"<<column.null_count()<<std::endl;
        // if (column.get_gdf_column()->valid)
        //     std::cout<<"copyGpuToCpu column has valid"<<std::endl;
        // else
        //     std::cout<<"copyGpuToCpu column has no valid"<<std::endl;

        try {
            if (isGdfString(*column.get_gdf_column())) {
                NVCategory * nvCategory = reinterpret_cast<NVCategory *>(
                    column.get_gdf_column()->dtype_info.category);

                NVStrings * nvStrings = nvCategory->to_strings();

                const std::size_t stringsLength = nvStrings->size();
                const std::size_t offsetsLength = stringsLength + 1;

                int * const lengthPerStrings = new int[stringsLength];
                // TODO: When implement null support, a null-string return -1 as
                // byte_count
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

                const std::size_t previousSize = result.size();

                // WARNING!!! When setting the size of result outside this function,
                // we are only getting the size for non-string columns. The size we
                // need for string columns is determined here inside the copyGpuToCpu
                // where it is resized again. THIS is a bad performance issue. This
                // needs to be addressed
                // TODO: Add to cuStrings functions to evaluate the strings and
                // offsets sizes before generate them and string array length
                result.resize(previousSize + totalSize);
                std::memcpy(
                    &result[binary_pointer], &stringsSize, sizeof(const std::size_t));
                std::memcpy(&result[binary_pointer + sizeof(const std::size_t)],
                            &offsetsSize,
                            sizeof(const std::size_t));
                std::memcpy(&result[binary_pointer + 2 * sizeof(const std::size_t)],
                            &stringsLength,
                            sizeof(const std::size_t));
                std::memcpy(&result[binary_pointer + 3 * sizeof(const std::size_t)],
                            stringsPointer,
                            stringsSize);
                std::memcpy(&result[binary_pointer + 3 * sizeof(const std::size_t) +
                                    stringsSize],
                            offsetsPointer,
                            offsetsSize);
                binary_pointer += totalSize;

                // TODO: remove pointers to map into `result` without bypass
                delete[] stringsPointer;
                delete[] offsetsPointer;
                delete[] lengthPerStrings;
                NVStrings::destroy(nvStrings);
            } else {
                std::size_t data_size = getDataCapacity(column.get_gdf_column());
                CUDA_TRY(cudaMemcpy(&result[binary_pointer],
                            column.data(),
                            data_size,
                            cudaMemcpyDeviceToHost));
                binary_pointer += data_size;

                std::size_t valid_size = getValidCapacity(column.get_gdf_column());
                CUDA_TRY(cudaMemcpy(&result[binary_pointer],
                            column.valid(),
                            valid_size,
                            cudaMemcpyDeviceToHost));
                binary_pointer += valid_size;                
            }
        } catch(const std::exception& e)  {
			std::cerr << "GpuFunctionsAdapter::copyGpuToCpu error => " << e.what() << '\n';			
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
          0, false);
    }

} // namespace adapter
} // namespace communication
} // namespace ral
