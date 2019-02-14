#pragma once

namespace ral {
namespace traits {

    template <typename T>
    struct get_dtype_from_type {
        static constexpr gdf_dtype type = GDF_invalid;
    };

    template <>
    struct get_dtype_from_type<int8_t> {
        static constexpr gdf_dtype type = GDF_INT8;
    };

    template <>
    struct get_dtype_from_type<int16_t> {
        static constexpr gdf_dtype type = GDF_INT16;
    };

    template <>
    struct get_dtype_from_type<int32_t> {
        static constexpr gdf_dtype type = GDF_INT32;
    };

    template <>
    struct get_dtype_from_type<int64_t> {
        static constexpr gdf_dtype type = GDF_INT64;
    };

    template <>
    struct get_dtype_from_type<float> {
        static constexpr gdf_dtype type = GDF_FLOAT32;
    };

    template <>
    struct get_dtype_from_type<double> {
        static constexpr gdf_dtype type = GDF_FLOAT64;
    };

    template <typename T>
    constexpr gdf_dtype get_dtype_from_type_v = get_dtype_from_type<T>::type;

} // namespace traits
} // namespace ral
