#include <algorithm>
#include <unordered_map>

#include "communication/adapter/GpuFunctionsAdapter.h"
#include "Traits/RuntimeTraits.h"
#include "GDFColumn.cuh"

#include <CodeTimer.h>
#include <blazing/metrics/chronometer.hpp>
#include <blazing/uss/conio.hpp>
#include "blazingdb/io/Library/Logging/Logger.h"

namespace ral {
namespace communication {
namespace adapter {

    class GpuFunctionsAdapter::StringsInfo {
    public:
        class StringInfo {
        public:
            explicit StringInfo(gdf_column_cpp & column)
                : nvStrings_{nullptr}, stringsLength_{0}, offsetsLength_{0},
                  stringsPointer_{nullptr}, offsetsPointer_{nullptr},
                  stringsSize_{0}, offsetsSize_{0}, totalSize_{0} {
                NVCategory * nvCategory = reinterpret_cast<NVCategory *>(
                    column.get_gdf_column()->dtype_info.category);

                if (!nvCategory) {
                    nvCategory = NVCategory::create_from_array(nullptr, 0);
                }

                nvStrings_ = nvCategory->to_strings();

                if (!nvStrings_) { return; }

                stringsLength_ = nvStrings_->size();
                offsetsLength_ = stringsLength_ + 1;

                int * const lengthPerStrings = new int[stringsLength_];
                // TODO: When implement null support, a null-string return -1 as
                // byte_count
                nvStrings_->byte_count(lengthPerStrings, false);

                stringsSize_ = std::accumulate(
                    lengthPerStrings, lengthPerStrings + stringsLength_, 0);
                offsetsSize_ = offsetsLength_ * sizeof(int);

                stringsPointer_ = new char[stringsSize_];
                offsetsPointer_ = new int[offsetsSize_];

                nvStrings_->create_offsets(
                    stringsPointer_, offsetsPointer_, nullptr, false);

                totalSize_ =
                    stringsSize_ + offsetsSize_ + 3 * sizeof(const std::size_t);

                delete[] lengthPerStrings;
            }

            ~StringInfo() {
                // TODO: remove pointers to map into `result` without bypass
                delete[] stringsPointer_;
                delete[] offsetsPointer_;
                NVStrings::destroy(nvStrings_);
            }

            std::size_t stringsLength() const noexcept {
                return stringsLength_;
            }

            std::size_t offsetsLength() const noexcept {
                return offsetsLength_;
            }

            char * stringsPointer() const noexcept { return stringsPointer_; }

            int * offsetsPointer() const noexcept { return offsetsPointer_; }

            std::size_t stringsSize() const noexcept { return stringsSize_; }

            std::size_t offsetsSize() const noexcept { return offsetsSize_; }

            std::size_t totalSize() const noexcept { return totalSize_; }

        private:
            NVStrings * nvStrings_;
            std::size_t stringsLength_;
            std::size_t offsetsLength_;
            char *      stringsPointer_;
            int *       offsetsPointer_;
            std::size_t stringsSize_;
            std::size_t offsetsSize_;
            std::size_t totalSize_;
        };

        explicit StringsInfo(std::vector<gdf_column_cpp> & columns) {
            for (gdf_column_cpp & column : columns) {
                gdf_column * gdfColumn = column.get_gdf_column();

                if (GpuFunctionsAdapter::isGdfString(*gdfColumn)) {
                    columnMap_.emplace(gdfColumn, new StringInfo{column});
                }
            }
        }

        ~StringsInfo() {
            std::for_each(
                columnMap_.cbegin(),
                columnMap_.cend(),
                [](const std::pair<gdf_column * const, StringInfo *> & pair) {
                    delete std::get<StringInfo *>(pair);
                });
        }

        std::size_t capacity() const noexcept {
            return std::accumulate(
                columnMap_.cbegin(),
                columnMap_.cend(),
                0,
                [](int & accumulator,
                   const std::pair<gdf_column * const, StringInfo *> & pair) {
                    return std::move(accumulator) +
                           std::get<StringInfo *>(pair)->totalSize();
                });
        }

        const StringInfo & At(gdf_column * gdfColumn) const {
            return *columnMap_.at(gdfColumn);
        }

    private:
        std::unordered_map<gdf_column *, StringInfo *> columnMap_;
    };

    const GpuFunctionsAdapter::StringsInfo *
    GpuFunctionsAdapter::createStringsInfo(
        std::vector<gdf_column_cpp> & columns) {
        const StringsInfo * stringsInfo = new StringsInfo{columns};
        return stringsInfo;
    }

    void
    GpuFunctionsAdapter::destroyStringsInfo(const StringsInfo * stringsInfo) {
        delete stringsInfo;
    }

    void GpuFunctionsAdapter::copyGpuToCpu(std::size_t &       binary_pointer,
                                           std::string &       result,
                                           gdf_column_cpp &    column,
                                           const StringsInfo * stringsInfo) {
        if (column.size() == 0) { return; }
        static CodeTimer timer;
        timer.reset();
        if (isGdfString(*column.get_gdf_column())) {
            using blazing::metrics::Chronometer;
            std::unique_ptr<Chronometer> chronometer =
                Chronometer::MakeStarted();


            const StringsInfo::StringInfo & stringInfo =
                stringsInfo->At(column.get_gdf_column());

            const std::size_t stringsSize   = stringInfo.stringsSize();
            const std::size_t offsetsSize   = stringInfo.offsetsSize();
            const std::size_t stringsLength = stringInfo.stringsLength();

            // WARNING!!! When setting the size of result outside this function,
            // we are only getting the size for non-string columns. The size we
            // need for string columns is determined here inside the
            // copyGpuToCpu where it is resized again. THIS is a bad performance
            // issue. This needs to be addressed
            // TODO: Add to cuStrings functions to evaluate the strings and
            // offsets sizes before generate them and string array length
            std::memcpy(&result[binary_pointer],
                        &stringsSize,
                        sizeof(const std::size_t));
            std::memcpy(&result[binary_pointer + sizeof(const std::size_t)],
                        &offsetsSize,
                        sizeof(const std::size_t));
            std::memcpy(&result[binary_pointer + 2 * sizeof(const std::size_t)],
                        &stringsLength,
                        sizeof(const std::size_t));
            std::memcpy(&result[binary_pointer + 3 * sizeof(const std::size_t)],
                        stringInfo.stringsPointer(),
                        stringsSize);
            std::memcpy(&result[binary_pointer + 3 * sizeof(const std::size_t) +
                                stringsSize],
                        stringInfo.offsetsPointer(),
                        offsetsSize);

            binary_pointer += stringInfo.totalSize();

            const std::uintmax_t elapsedTime = chronometer->Elapsed();

            using blazing::uss::conio::Console;
            Console console;

            console.SetColor(Console::kGreen)
                .Write("String column serializing time for \"" +
                       std::string{column.get_gdf_column()->col_name} +
                       "\": " + std::to_string(elapsedTime))
                .SetColor(Console::kNone)
                .EndLine();
          
            Library::Logging::Logger().logInfo("-> copyGpuToCpu:GdfString " + std::to_string(timer.getDuration()) + " ms");
        } else {
            std::size_t data_size = getDataCapacity(column.get_gdf_column());
            CheckCudaErrors(cudaMemcpyAsync(&result[binary_pointer],
                                            column.data(),
                                            data_size,
                                            cudaMemcpyDeviceToHost,
                                            0));
            binary_pointer += data_size;

            std::size_t valid_size = getValidCapacity(column.get_gdf_column());
            CheckCudaErrors(cudaMemcpyAsync(&result[binary_pointer],
                                            column.valid(),
                                            valid_size,
                                            cudaMemcpyDeviceToHost,
                                            0));
            binary_pointer += valid_size;
          
            cudaStreamSynchronize(0);

            Library::Logging::Logger().logInfo("-> copyGpuToCpu:data_valid " + std::to_string(timer.getDuration()) + " ms");
        }
    }

    std::size_t GpuFunctionsAdapter::getDataCapacity(gdf_column* column) {
        return ral::traits::get_data_size_in_bytes(column);
    }

    std::size_t GpuFunctionsAdapter::getValidCapacity(gdf_column* column) {
        return column->null_count > 0 ? ral::traits::get_bitmask_size_in_bytes(column->size) : 0;
    }

    std::size_t GpuFunctionsAdapter::getStringsCapacity(const StringsInfo *stringsInfo) {
      return stringsInfo->capacity();
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
