#include <gtest/gtest.h>
#include <vector>
#include <bitset>
#include <algorithm>
#include "Traits/RuntimeTraits.h"
#include "cuDF/column_slice/column_cpp_slice.h"
#include "tests/utilities/gdf_column_cpp_utilities.h"

namespace {

template <std::size_t SIZE>
std::vector<std::uint8_t> slice_cpu_valids(const gdf_column_cpp& input_column, std::size_t start_bit, std::size_t bits_length) {
    // get input valids
    std::vector<std::uint8_t> input_valid = ral::test::get_column_valid(input_column.get_gdf_column());

    // bitset data
    std::bitset<SIZE> bitset{};

    // populate data into bitset
    std::size_t bit_index {0};
    auto put_data_bitset = [&bitset, &bit_index](std::uint8_t value) {
        for (int k = 0; k < 8; ++k) {
            if (SIZE <= bit_index) {
                break;
            }
            bitset[bit_index] = value & 1;
            value >>= 1;
            bit_index++;
        }
    };

    std::for_each(input_valid.begin(), input_valid.end(), put_data_bitset);

    // perform shift operation
    bitset >>= start_bit;

    // calculate result byte size with padding
    const auto const_byte_size = ral::traits::BYTE_SIZE_IN_BITS;
    std::size_t result_byte_size = (bits_length / const_byte_size) + ((bits_length % const_byte_size) ? 1 : 0);

    const auto align_size = ral::traits::BITMASK_SIZE_IN_BYTES;
    std::size_t result_byte_size_padding = ((result_byte_size + (align_size - 1)) / align_size) * align_size;

    // extract data from bitset
    bit_index = 0;
    auto get_data_bitset = [&bitset, &bit_index, &bits_length]() {
        std::uint8_t value = 0;
        for (int k = 0; k < ral::traits::BYTE_SIZE_IN_BITS; ++k) {
            if (bits_length <= bit_index) {
                return value;
            }
            std::uint8_t tmp = bitset[bit_index];
            value |= (tmp << k);
            bit_index++;
        }
        return value;
    };

    std::vector<std::uint8_t> result(result_byte_size_padding, 0);
    std::generate_n(result.begin(), result_byte_size_padding, get_data_bitset);

    // done
    return result;
}

/**
 * It makes 128 tests, where the start bit index position goes from 0 to 128.
 */
struct ColumnSliceInitialIndexTest : public ::testing::TestWithParam<std::size_t>
{
    ColumnSliceInitialIndexTest() {
        input_column_ = ral::test::create_gdf_column_cpp(column_size_, dtype_);
    }

    gdf_dtype dtype_ {GDF_INT64};
    gdf_column_cpp input_column_;

    static constexpr gdf_size_type bit_length {32};
    static constexpr std::size_t column_size_{160};
};

TEST_P(ColumnSliceInitialIndexTest, BodyTest) {
    // get start bit index
    const gdf_size_type bit_start = GetParam();

    // create gdf_column_cpp output
    gdf_column_cpp output_column = ral::test::create_null_gdf_column_cpp(bit_length, dtype_);

    // execute slice function
    auto error = ral::cudf::column_cpp_valid_slice(&output_column, &input_column_, bit_start, bit_length);
    ASSERT_EQ(error, GDF_SUCCESS);

    // get input and output valids
    const auto input_valid = slice_cpu_valids<column_size_>(input_column_, bit_start, bit_length);
    const auto output_valid = ral::test::get_column_valid(output_column.get_gdf_column());

    // verify
    ASSERT_EQ(input_valid.size(), output_valid.size());
    for (std::size_t i = 0; i < input_valid.size(); ++i) {
        ASSERT_EQ(input_valid[i], output_valid[i]);
    }
}

INSTANTIATE_TEST_CASE_P(InitialIndexPosition, ColumnSliceInitialIndexTest, ::testing::Range<std::size_t>(0, 128));

/**
 * It makes 128 tests, where the bit length goes from 0 bits to 128 bits.
 */
struct ColumnSliceBitLengthTest : public ::testing::TestWithParam<std::size_t>
{
    ColumnSliceBitLengthTest() {
      input_column_ = ral::test::create_gdf_column_cpp(column_size_, dtype_);
    }

    gdf_dtype dtype_ {GDF_INT32};
    gdf_column_cpp input_column_;

    static constexpr gdf_size_type bit_start {32};
    static constexpr std::size_t column_size_{160};
};

TEST_P(ColumnSliceBitLengthTest, BodyTest) {
    // get start bit index
    const gdf_size_type bit_length = GetParam();

    // create gdf_column_cpp output
    gdf_column_cpp output_column = ral::test::create_null_gdf_column_cpp(bit_length, dtype_);

    // execute slice function
    auto error = ral::cudf::column_cpp_valid_slice(&output_column, &input_column_, bit_start, bit_length);
    ASSERT_EQ(error, GDF_SUCCESS);

    // get input and output valids
    const auto input_valid = slice_cpu_valids<column_size_>(input_column_, bit_start, bit_length);
    const auto output_valid = ral::test::get_column_valid(output_column.get_gdf_column());

    // verify
    ASSERT_EQ(input_valid.size(), output_valid.size());
    for (std::size_t i = 0; i < input_valid.size(); ++i) {
        ASSERT_EQ(input_valid[i], output_valid[i]);
    }
}

INSTANTIATE_TEST_CASE_P(BitLength, ColumnSliceBitLengthTest, ::testing::Range<std::size_t>(0, 128));

} // namespace
