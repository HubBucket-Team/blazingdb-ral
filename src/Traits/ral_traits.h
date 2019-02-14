#pragma once

namespace ral {
namespace traits {

    template <gdf_dtype T>
    struct get_type;

    template <>
    struct get_type<GDF_INT8> {
        using type = std::int8_t;
    };

    template <>
    struct get_type<GDF_INT16> {
        using type = std::int16_t;
    };

    template <>
    struct get_type<GDF_INT32> {
        using type = std::int32_t;
    };

    template <>
    struct get_type<GDF_INT64> {
        using type = std::int64_t;
    };

    template <>
    struct get_type<GDF_FLOAT32> {
        using type = float;
    };

    template <>
    struct get_type<GDF_FLOAT64> {
        using type = double;
    };

    template <>
    struct get_type<GDF_DATE32> {
        using type = std::int32_t;
    };

    template <>
    struct get_type<GDF_DATE64> {
        using type = std::int64_t;
    };

    template <>
    struct get_type<GDF_TIMESTAMP> {
        using type = std::int64_t;
    };

    template <gdf_dtype E>
    using get_type_t = typename get_type<E>::type;

} // namespace traits
} // namespace ral
