#include "gtest/gtest.h"
#include "GDFColumn.cuh"
#include "GDFCounter.cuh"

namespace {

    struct GdfColumnCppTest : public testing::Test {
        GdfColumnCppTest() {
            counter_instance = GDFRefCounter::getInstance();
        }

        ~GdfColumnCppTest() {
        }

        void SetUp() override {
        }

        void TearDown() override {
        }

        GDFRefCounter* counter_instance;
    };


    TEST_F(GdfColumnCppTest, AssignOperatorInGdfCounter) {
        gdf_column* gdf_col_1{};
        gdf_column* gdf_col_2{};
        gdf_column* gdf_col_3{};

        ASSERT_EQ(counter_instance->get_map_size(), 0);

        {
            // initialization
            gdf_column_cpp cpp_col_1;
            cpp_col_1.create_gdf_column(GDF_INT32, 16, nullptr, 4, "sample");
            gdf_col_1 = cpp_col_1.get_gdf_column();

            gdf_column_cpp cpp_col_2;
            cpp_col_2.create_gdf_column(GDF_INT64, 32, nullptr, 8, "sample");
            gdf_col_2 = cpp_col_2.get_gdf_column();

            gdf_column_cpp cpp_col_3;
            cpp_col_3.create_gdf_column(GDF_INT64, 32, nullptr, 8, "sample");
            gdf_col_3 = cpp_col_3.get_gdf_column();

            ASSERT_EQ(counter_instance->get_map_size(), 3);
            ASSERT_EQ(counter_instance->column_ref_value(gdf_col_1), 1);
            ASSERT_EQ(counter_instance->column_ref_value(gdf_col_2), 1);
            ASSERT_EQ(counter_instance->column_ref_value(gdf_col_3), 1);

            // test assign operator
            cpp_col_2 = cpp_col_1;

            ASSERT_EQ(counter_instance->get_map_size(), 2);
            ASSERT_EQ(counter_instance->column_ref_value(gdf_col_1), 2);
            ASSERT_EQ(counter_instance->contains_column(gdf_col_2), false);
            ASSERT_EQ(counter_instance->column_ref_value(gdf_col_3), 1);

            // test assign operator on equal gdf_column_cpp
            cpp_col_1 = cpp_col_2;

            ASSERT_EQ(counter_instance->get_map_size(), 2);
            ASSERT_EQ(counter_instance->column_ref_value(gdf_col_1), 2);
            ASSERT_EQ(counter_instance->contains_column(gdf_col_2), false);
            ASSERT_EQ(counter_instance->column_ref_value(gdf_col_3), 1);

            // test assign operator again
            cpp_col_1 = cpp_col_3;

            ASSERT_EQ(counter_instance->get_map_size(), 2);
            ASSERT_EQ(counter_instance->column_ref_value(gdf_col_1), 1);
            ASSERT_EQ(counter_instance->contains_column(gdf_col_2), false);
            ASSERT_EQ(counter_instance->column_ref_value(gdf_col_3), 2);
        }

        ASSERT_EQ(counter_instance->get_map_size(), 0);
        ASSERT_TRUE(counter_instance->contains_column(gdf_col_1) == false);
        ASSERT_TRUE(counter_instance->contains_column(gdf_col_2) == false);
        ASSERT_TRUE(counter_instance->contains_column(gdf_col_3) == false);
    }


    // void gdf_column_cpp::create_gdf_column(gdf_dtype type,
    //                                        size_t num_values,
    //                                        void * input_data,
    //                                        gdf_valid_type * host_valids,
    //                                        size_t width_per_value,
    //                                        const std::string &column_name)
    TEST_F(GdfColumnCppTest, CreateGdfColumnCppTypeOne_DoubleCreation) {
        gdf_column* gdf_col_1{};
        gdf_column* gdf_col_2{};
        ASSERT_EQ(counter_instance->get_map_size(), 0);

        {
            // initialize
            gdf_column_cpp cpp_col;
            cpp_col.create_gdf_column(GDF_INT32,
                                      32,
                                      nullptr,
                                      nullptr,
                                      4,
                                      "sample 1");
            gdf_col_1 = cpp_col.get_gdf_column();

            ASSERT_EQ(counter_instance->get_map_size(), 1);
            ASSERT_EQ(counter_instance->column_ref_value(gdf_col_1), 1);
            ASSERT_EQ(counter_instance->contains_column(gdf_col_2), false);

            // create again - note: os could reuse the pointer
            cpp_col.create_gdf_column(GDF_INT64,
                                      64,
                                      nullptr,
                                      nullptr,
                                      8,
                                      "sample 2");
            gdf_col_2 = cpp_col.get_gdf_column();

            ASSERT_EQ(counter_instance->get_map_size(), 1);
            if (gdf_col_1 == gdf_col_2) {
                ASSERT_EQ(counter_instance->column_ref_value(gdf_col_1), 1);
            }
            else {
                ASSERT_EQ(counter_instance->contains_column(gdf_col_1), false);
                ASSERT_EQ(counter_instance->column_ref_value(gdf_col_2), 1);
            }
        }

        ASSERT_EQ(counter_instance->get_map_size(), 0);
        ASSERT_TRUE(counter_instance->contains_column(gdf_col_1) == false);
        ASSERT_TRUE(counter_instance->contains_column(gdf_col_2) == false);
    }

}