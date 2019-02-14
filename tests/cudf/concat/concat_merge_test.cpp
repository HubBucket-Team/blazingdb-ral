#include "gtest/gtest.h"
#include "DataFrame.h"
#include "src/Traits/ral_traits.h"
#include "cuDF/concat/concat_merge.h"
#include "cudf/concat/database_path.h"
#include "database/random_numbers/data_handler.h"
#include "gdf_wrapper/utilities/cudf_utils.h"

namespace {

    struct ConcatMergeTest : public testing::Test {
        ConcatMergeTest() {
            bdb::test::DataHandler handler(bdb::test::database_path);
            data = handler.readData();
            meta = handler.getMeta();
        }

        ~ConcatMergeTest() {
        }

        void SetUp() override {
        }

        void TearDown() override {
        }

        struct Sequence {
            int index{};
            int size{};
        };

        template <typename T>
        auto create_data_vector(const Sequence& sequence) -> std::vector<std::vector<T>> {
            std::vector<std::vector<T>> column_data;
            std::transform(std::next(data.begin(), sequence.index),
                           std::next(data.begin(), sequence.index + sequence.size),
                           std::back_inserter(column_data),
                           [](std::vector<bdb::test::Data>& aux_data) {
                               std::vector<T> out_data;
                               std::transform(aux_data.begin(),
                                              aux_data.end(),
                                              std::back_inserter(out_data),
                                              [](bdb::test::Data& core_data) {
                                                  return core_data.data;
                                              });
                               return out_data;
                           });

            return column_data;
        }

        auto create_mask_vector(const Sequence& sequence) -> std::vector<std::vector<gdf_valid_type>> {
            std::vector<std::vector<gdf_valid_type>> column_mask;
            for (std::size_t i = 0; i < sequence.size; ++i) {
                std::vector<gdf_valid_type> aux_data;
                for (std::size_t j = 0; j < meta.size_column; j += GDF_VALID_BITSIZE) {
                    gdf_valid_type aux = 0;
                    for (std::size_t k = 0; k < GDF_VALID_BITSIZE; ++k) {
                        aux += data[sequence.index + i][j + k].mask;
                        if (k < (GDF_VALID_BITSIZE - 1)) {
                            aux <<= 1;
                        }
                    }
                    aux_data.emplace_back(aux);
                }
                column_mask.emplace_back(aux_data);
            }

            return column_mask;
        }

        template <typename T>
        void verify_data(blazing_frame& frame, int index, const Sequence& sequence) {
            auto column_data = create_data_vector<T>(sequence);
            auto gdf_column = frame.get_columns()[0][index];

            std::size_t total_size{};
            for (const auto& vector : column_data) {
                total_size += vector.size();
            }
            ASSERT_EQ(gdf_column.size(), total_size);

            std::vector<T> output_data(gdf_column.size());
            cudaMemcpy(output_data.data(), gdf_column.data(), sizeof(T) * gdf_column.size(), cudaMemcpyDeviceToHost);

            std::size_t counter{};
            for (const auto& vector : column_data) {
                for (const auto& value : vector) {
                    ASSERT_EQ(output_data[counter++], value);
                }
            }
        }

        void verify_mask(blazing_frame& frame, int index, const Sequence& sequence) {
            auto column_mask = create_mask_vector(sequence);
            auto gdf_column = frame.get_columns()[0][index];

            std::size_t total_size{};
            for (const auto& vector : column_mask) {
                total_size += vector.size();
            }
            total_size = ((total_size + 3) / 4) * 4;
            ASSERT_EQ(gdf_column.get_valid_size(), total_size);

            std::vector<gdf_valid_type> output_mask(gdf_column.get_valid_size());
            cudaMemcpy(output_mask.data(), gdf_column.valid(), gdf_column.get_valid_size(), cudaMemcpyDeviceToHost);

            std::size_t counter{};
            for (const auto& vector : column_mask) {
                for (const auto& value : vector) {
                    ASSERT_EQ(output_mask[counter++], value);
                }
            }
        }

        template <typename T>
        void createInputFrame(std::vector<blazing_frame>& frame, const Sequence& sequence) {
            // create data vector
            auto column_data = create_data_vector<T>(sequence);

            // create valid vector
            auto column_mask = create_mask_vector(sequence);

            // create gdf_columns
            std::vector<gdf_column_cpp> columns(sequence.size);
            for (std::size_t i = 0; i < sequence.size; ++i) {
                columns[i].create_gdf_column(ral::traits::get_dtype_from_type_v<T>,
                                             meta.size_column,
                                             column_data[i].data(),
                                             column_mask[i].data(),
                                             sizeof(T),
                                             "");
            }

            // add gdf_columns to frame
            frame[0].add_table(columns);
        }

        bdb::test::Meta meta;
        std::vector<std::vector<bdb::test::Data>> data;
    };


    TEST_F(ConcatMergeTest, Int8_WithTwoSequence) {
        // create sequences
        std::vector<Sequence> sequences;
        sequences.emplace_back(Sequence{ .index = 0, .size = 1 });
        sequences.emplace_back(Sequence{ .index = 1, .size = 2 });

        // create input frame
        std::vector<blazing_frame> input_frame(1);

        // populate input frame
        for (const auto& sequence : sequences) {
            createInputFrame<int8_t>(input_frame, sequence);
        }

        // create output frame
        blazing_frame output_frame;

        // perform test - concat merge
        auto error = cudf::concat::concat_merge(input_frame, output_frame);
        ASSERT_TRUE(error == GDF_SUCCESS);

        // verify test
        for (int k = 0; k < (int)sequences.size(); ++k) {
            verify_data<int8_t>(output_frame, k, sequences[k]);
            verify_mask(output_frame, k, sequences[k]);
        }
    }


    TEST_F(ConcatMergeTest, Int16_WithTwoSequence) {
        // create sequences
        std::vector<Sequence> sequences;
        sequences.emplace_back(Sequence{ .index = 3, .size = 3 });
        sequences.emplace_back(Sequence{ .index = 6, .size = 4 });

        // create input frame
        std::vector<blazing_frame> input_frame(1);

        // populate input frame
        for (const auto& sequence : sequences) {
            createInputFrame<int16_t>(input_frame, sequence);
        }

        // create output frame
        blazing_frame output_frame;

        // perform test - concat merge
        auto error = cudf::concat::concat_merge(input_frame, output_frame);
        ASSERT_TRUE(error == GDF_SUCCESS);

        // verify test
        for (int k = 0; k < (int)sequences.size(); ++k) {
            verify_data<int16_t>(output_frame, k, sequences[k]);
            verify_mask(output_frame, k, sequences[k]);
        }
    }


    TEST_F(ConcatMergeTest, Int32_WithThreeSequence) {
        // create sequences
        std::vector<Sequence> sequences;
        sequences.emplace_back(Sequence{ .index = 0, .size = 5 });
        sequences.emplace_back(Sequence{ .index = 5, .size = 3 });
        sequences.emplace_back(Sequence{ .index = 8, .size = 2 });

        // create input frame
        std::vector<blazing_frame> input_frame(1);

        // populate input frame
        for (const auto& sequence : sequences) {
            createInputFrame<int32_t>(input_frame, sequence);
        }

        // create output frame
        blazing_frame output_frame;

        // perform test - concat merge
        auto error = cudf::concat::concat_merge(input_frame, output_frame);
        ASSERT_TRUE(error == GDF_SUCCESS);

        // verify test
        for (int k = 0; k < (int)sequences.size(); ++k) {
            verify_data<int32_t>(output_frame, k, sequences[k]);
            verify_mask(output_frame, k, sequences[k]);
        }
    }


    TEST_F(ConcatMergeTest, Int64_WithFourSequence) {
        // create sequences
        std::vector<Sequence> sequences;
        sequences.emplace_back(Sequence{ .index = 0, .size = 1 });
        sequences.emplace_back(Sequence{ .index = 1, .size = 2 });
        sequences.emplace_back(Sequence{ .index = 3, .size = 3 });
        sequences.emplace_back(Sequence{ .index = 6, .size = 4 });

        // create input frame
        std::vector<blazing_frame> input_frame(1);

        // populate input frame
        for (const auto& sequence : sequences) {
            createInputFrame<int64_t>(input_frame, sequence);
        }

        // create output frame
        blazing_frame output_frame;

        // perform test - concat merge
        auto error = cudf::concat::concat_merge(input_frame, output_frame);
        ASSERT_TRUE(error == GDF_SUCCESS);

        // verify test
        for (int k = 0; k < (int)sequences.size(); ++k) {
            verify_data<int64_t>(output_frame, k, sequences[k]);
            verify_mask(output_frame, k, sequences[k]);
        }
    }


    TEST_F(ConcatMergeTest, Int8_Int16_Float32_Float64) {
        // create sequences
        std::vector<Sequence> sequences;
        sequences.emplace_back(Sequence{ .index = 0, .size = 1 });
        sequences.emplace_back(Sequence{ .index = 1, .size = 2 });
        sequences.emplace_back(Sequence{ .index = 3, .size = 3 });
        sequences.emplace_back(Sequence{ .index = 6, .size = 4 });

        // create input frame
        std::vector<blazing_frame> input_frame(1);

        // populate input frame
        createInputFrame<int8_t>(input_frame, sequences[0]);
        createInputFrame<int16_t>(input_frame, sequences[1]);
        createInputFrame<float>(input_frame, sequences[2]);
        createInputFrame<double>(input_frame, sequences[3]);

        // create output frame
        blazing_frame output_frame;

        // perform test - concat merge
        auto error = cudf::concat::concat_merge(input_frame, output_frame);
        ASSERT_TRUE(error == GDF_SUCCESS);

        // verify test
        verify_data<int8_t>(output_frame, 0, sequences[0]);
        verify_mask(output_frame, 0, sequences[0]);
        verify_data<int16_t>(output_frame, 1, sequences[1]);
        verify_mask(output_frame, 1, sequences[1]);
        verify_data<float>(output_frame, 2, sequences[2]);
        verify_mask(output_frame, 2, sequences[2]);
        verify_data<double>(output_frame, 3, sequences[3]);
        verify_mask(output_frame, 3, sequences[3]);
    }


    TEST_F(ConcatMergeTest, DestroySelected) {
        // create sequences
        std::vector<Sequence> sequences;
        sequences.emplace_back(Sequence{ .index = 2, .size = 1 });
        sequences.emplace_back(Sequence{ .index = 6, .size = 2 });

        // create input frame
        std::vector<blazing_frame> input_frame(1);

        // populate input frame
        for (const auto& sequence : sequences) {
            createInputFrame<int8_t>(input_frame, sequence);
        }

        // create output frame
        blazing_frame output_frame;

        // perform test - concat merge
        auto error = cudf::concat::concat_merge(input_frame, output_frame, true);
        ASSERT_EQ(error, GDF_SUCCESS);
        ASSERT_EQ(input_frame.size(), 0);

        // verify test
        for (int k = 0; k < (int)sequences.size(); ++k) {
            verify_data<int8_t>(output_frame, k, sequences[k]);
            verify_mask(output_frame, k, sequences[k]);
        }
    }


    TEST_F(ConcatMergeTest, InputError_EmptyBlazingFrame) {
        // create input frame
        std::vector<blazing_frame> input_frame;

        // create output frame
        blazing_frame output_frame;

        // perform test - concat merge
        auto error = cudf::concat::concat_merge(input_frame, output_frame);

        // verify
        ASSERT_EQ(error, GDF_DATASET_EMPTY);
        ASSERT_EQ(output_frame.get_columns().size(), 0);
    }


    TEST_F(ConcatMergeTest, InputError_DifferentTypeBlazingFrame) {
        // create input frame
        std::vector<blazing_frame> input_frame(1);

        // set input frame
        {
            Sequence sequence1{ .index = 0, .size = 1 };
            Sequence sequence2{ .index = 1, .size = 1 };

            // create data vector
            auto column_data_1 = create_data_vector<int8_t>(sequence1);
            auto column_data_2 = create_data_vector<int64_t>(sequence2);

            // create valid vector
            auto column_mask_1 = create_mask_vector(sequence1);
            auto column_mask_2 = create_mask_vector(sequence2);

            // create gdf_columns
            std::vector<gdf_column_cpp> columns(2);
            columns[0].create_gdf_column(ral::traits::get_dtype_from_type_v<int8_t>,
                                         meta.size_column,
                                         column_data_1[0].data(),
                                         column_mask_1[0].data(),
                                         sizeof(int8_t),
                                         "");

            columns[1].create_gdf_column(ral::traits::get_dtype_from_type_v<int64_t>,
                                         meta.size_column,
                                         column_data_2[0].data(),
                                         column_mask_2[0].data(),
                                         sizeof(int64_t),
                                         "");

            // add gdf_columns to frame
            input_frame[0].add_table(columns);
        }

        // create output frame
        blazing_frame output_frame;

        // perform test - concat merge
        auto error = cudf::concat::concat_merge(input_frame, output_frame);

        // verify
        ASSERT_EQ(error, GDF_DTYPE_MISMATCH);
        ASSERT_EQ(output_frame.get_columns().size(), 0);
    }

} // namespace
