
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <CalciteExpressionParsing.h>
#include <CalciteInterpreter.h>
#include <DataFrame.h>
#include <StringUtil.h>
#include <gdf/gdf.h>
#include <gtest/gtest.h>
#include <GDFColumn.cuh>
#include <GDFCounter.cuh>
#include <Utils.cuh>

#include "gdf/library/api.h"
using namespace gdf::library;

struct EvaluateQueryTest : public ::testing::Test {
  struct InputTestItem {
    std::string query;
    std::string logicalPlan;
    gdf::library::TableGroup tableGroup;
    gdf::library::Table resultTable;
  };

  void CHECK_RESULT(gdf::library::Table& computed_solution,
                    gdf::library::Table& reference_solution) {
    computed_solution.print(std::cout);
    reference_solution.print(std::cout);

    for (size_t index = 0; index < reference_solution.size(); index++) {
      const auto& reference_column = reference_solution[index];
      const auto& computed_column = computed_solution[index];
      auto a = reference_column.to_string();
      auto b = computed_column.to_string();
      EXPECT_EQ(a, b);
    }
  }
};

// AUTO GENERATED UNIT TESTS
TEST_F(EvaluateQueryTest, TEST_00) {
  auto input = InputTestItem{
      .query = "select r_regionkey from main.region order by r_regionkey desc",
      .logicalPlan =
          "LogicalSort(sort0=[$0], dir0=[DESC])\n  "
          "LogicalProject(r_regionkey=[$0])\n    "
          "EnumerableTableScan(table=[[main, region]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.region",
               {{"r_regionkey", Literals<GDF_INT8>{0, 1, 2, 3, 4}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{"ResultSet",
                              {{"GDF_INT8", Literals<GDF_INT8>{4, 3, 2, 1, 0}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_01) {
  auto input = InputTestItem{
      .query = "select r_regionkey from main.region order by r_regionkey",
      .logicalPlan =
          "LogicalSort(sort0=[$0], dir0=[ASC])\n  "
          "LogicalProject(r_regionkey=[$0])\n    "
          "EnumerableTableScan(table=[[main, region]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.region",
               {{"r_regionkey", Literals<GDF_INT8>{0, 1, 2, 3, 4}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{"ResultSet",
                              {{"GDF_INT8", Literals<GDF_INT8>{0, 1, 2, 3, 4}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_02) {
  auto input = InputTestItem{
      .query =
          "select c_custkey, c_acctbal from main.customer order by c_acctbal",
      .logicalPlan =
          "LogicalSort(sort0=[$1], dir0=[ASC])\n  "
          "LogicalProject(c_custkey=[$0], c_acctbal=[$2])\n    "
          "EnumerableTableScan(table=[[main, customer]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.customer",
               {{"c_custkey",
                 Literals<GDF_INT32>{
                     1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,
                     13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
                     25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,
                     37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
                     49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,
                     61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
                     73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,
                     85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
                     97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108,
                     109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                     121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
                     133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
                     145, 146, 147, 148, 149, 150}},
                {"c_nationkey",
                 Literals<GDF_INT32>{
                     15, 13, 1,  4,  3,  20, 18, 17, 8,  5,  23, 13, 3,  1,
                     23, 10, 2,  6,  18, 22, 8,  3,  3,  13, 12, 22, 3,  8,
                     0,  1,  23, 15, 17, 15, 17, 21, 8,  12, 2,  3,  10, 5,
                     19, 16, 9,  6,  2,  0,  10, 6,  12, 11, 15, 4,  10, 10,
                     21, 13, 1,  12, 17, 7,  21, 3,  23, 22, 9,  12, 9,  22,
                     7,  2,  0,  4,  18, 0,  17, 9,  15, 0,  20, 18, 22, 11,
                     5,  0,  23, 16, 14, 16, 8,  2,  7,  9,  15, 8,  17, 12,
                     15, 20, 2,  19, 9,  10, 10, 1,  15, 5,  16, 10, 22, 19,
                     12, 14, 8,  16, 24, 18, 7,  12, 17, 3,  5,  18, 19, 22,
                     21, 4,  7,  9,  11, 4,  17, 11, 19, 7,  16, 5,  9,  4,
                     1,  9,  16, 1,  13, 3,  18, 11, 19, 18}},
                {"c_acctbal",
                 Literals<GDF_FLOAT32>{
                     711.56,  121.65,  7498.12, 2866.83, 794.47,  7638.57,
                     9561.95, 6819.74, 8324.07, 2753.54, -272.6,  3396.49,
                     3857.34, 5266.3,  2788.52, 4681.03, 6.34,    5494.43,
                     8914.71, 7603.4,  1428.25, 591.98,  3332.02, 9255.67,
                     7133.7,  5182.05, 5679.84, 1007.18, 7618.27, 9321.01,
                     5236.89, 3471.53, -78.56,  8589.7,  1228.24, 4987.27,
                     -917.75, 6345.11, 6264.31, 1335.3,  270.95,  8727.01,
                     9904.28, 7315.94, 9983.38, 5744.59, 274.58,  3792.5,
                     4573.94, 4266.13, 855.87,  5630.28, 4113.64, 868.9,
                     4572.11, 6530.86, 4151.93, 6478.46, 3458.6,  2741.87,
                     1536.24, 595.61,  9331.13, -646.64, 8795.16, 242.77,
                     8166.59, 6853.37, 1709.28, 4867.52, -611.19, -362.86,
                     4288.5,  2764.43, 6684.1,  5745.33, 1738.87, 7136.97,
                     5121.28, 7383.53, 2023.71, 9468.34, 6463.51, 5174.71,
                     3386.64, 3306.32, 6327.54, 8031.44, 1530.76, 7354.23,
                     4643.14, 1182.91, 2182.52, 5500.11, 5327.38, 6323.92,
                     2164.48, -551.37, 4088.65, 9889.89, 7470.96, 8462.17,
                     2757.45, -588.38, 9091.82, 3288.42, 2514.15, 2259.38,
                     -716.1,  7462.99, 6505.26, 2953.35, 2912.0,  1027.46,
                     7508.92, 8403.99, 3950.83, 3582.37, 3930.35, 363.75,
                     6428.32, 7865.46, 5897.83, 1842.49, -234.12, 1001.39,
                     9280.71, -986.96, 9127.27, 5073.58, 8595.53, 162.57,
                     2314.67, 4608.9,  8732.91, -842.39, 7838.3,  430.59,
                     7897.78, 9963.15, 6706.14, 2209.81, 2186.5,  6417.31,
                     9748.93, 3328.68, 8071.4,  2135.6,  8959.65, 3849.48}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32",
                Literals<GDF_INT32>{
                    128, 37,  136, 109, 64,  71,  104, 98,  72,  11,  125, 33,
                    17,  2,   132, 66,  41,  47,  120, 138, 22,  62,  1,   5,
                    51,  54,  126, 28,  114, 92,  35,  40,  21,  89,  61,  69,
                    77,  124, 81,  148, 97,  93,  143, 142, 108, 133, 107, 60,
                    10,  103, 74,  15,  4,   113, 112, 106, 86,  146, 23,  85,
                    12,  59,  32,  118, 48,  150, 13,  119, 117, 99,  53,  57,
                    50,  73,  55,  49,  134, 91,  16,  70,  36,  130, 79,  84,
                    26,  31,  14,  95,  18,  94,  52,  27,  46,  76,  123, 39,
                    96,  87,  38,  144, 121, 83,  58,  111, 56,  75,  141, 8,
                    68,  25,  78,  44,  90,  80,  110, 101, 3,   115, 20,  29,
                    6,   137, 122, 139, 88,  147, 67,  9,   116, 102, 34,  131,
                    42,  135, 65,  19,  149, 105, 129, 24,  127, 30,  63,  82,
                    7,   145, 100, 43,  140, 45}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{
                    -986.96, -917.75, -842.39, -716.1,  -646.64, -611.19,
                    -588.38, -551.37, -362.86, -272.6,  -234.12, -78.56,
                    6.34,    121.65,  162.57,  242.77,  270.95,  274.58,
                    363.75,  430.59,  591.98,  595.61,  711.56,  794.47,
                    855.87,  868.9,   1001.39, 1007.18, 1027.46, 1182.91,
                    1228.24, 1335.3,  1428.25, 1530.76, 1536.24, 1709.28,
                    1738.87, 1842.49, 2023.71, 2135.6,  2164.48, 2182.52,
                    2186.5,  2209.81, 2259.38, 2314.67, 2514.15, 2741.87,
                    2753.54, 2757.45, 2764.43, 2788.52, 2866.83, 2912.0,
                    2953.35, 3288.42, 3306.32, 3328.68, 3332.02, 3386.64,
                    3396.49, 3458.6,  3471.53, 3582.37, 3792.5,  3849.48,
                    3857.34, 3930.35, 3950.83, 4088.65, 4113.64, 4151.93,
                    4266.13, 4288.5,  4572.11, 4573.94, 4608.9,  4643.14,
                    4681.03, 4867.52, 4987.27, 5073.58, 5121.28, 5174.71,
                    5182.05, 5236.89, 5266.3,  5327.38, 5494.43, 5500.11,
                    5630.28, 5679.84, 5744.59, 5745.33, 5897.83, 6264.31,
                    6323.92, 6327.54, 6345.11, 6417.31, 6428.32, 6463.51,
                    6478.46, 6505.26, 6530.86, 6684.1,  6706.14, 6819.74,
                    6853.37, 7133.7,  7136.97, 7315.94, 7354.23, 7383.53,
                    7462.99, 7470.96, 7498.12, 7508.92, 7603.4,  7618.27,
                    7638.57, 7838.3,  7865.46, 7897.78, 8031.44, 8071.4,
                    8166.59, 8324.07, 8403.99, 8462.17, 8589.7,  8595.53,
                    8727.01, 8732.91, 8795.16, 8914.71, 8959.65, 9091.82,
                    9127.27, 9255.67, 9280.71, 9321.01, 9331.13, 9468.34,
                    9561.95, 9748.93, 9889.89, 9904.28, 9963.15, 9983.38}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_03) {
  auto input = InputTestItem{
      .query =
          "select c_custkey, c_nationkey, c_acctbal from main.customer order "
          "by c_acctbal",
      .logicalPlan =
          "LogicalSort(sort0=[$2], dir0=[ASC])\n  "
          "LogicalProject(c_custkey=[$0], c_nationkey=[$1], c_acctbal=[$2])\n  "
          "  EnumerableTableScan(table=[[main, customer]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.customer",
               {{"c_custkey",
                 Literals<GDF_INT32>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50}},
                {"c_nationkey",
                 Literals<GDF_INT32>{15, 13, 1,  4,  3,  20, 18, 17, 8,  5,
                                     23, 13, 3,  1,  23, 10, 2,  6,  18, 22,
                                     8,  3,  3,  13, 12, 22, 3,  8,  0,  1,
                                     23, 15, 17, 15, 17, 21, 8,  12, 2,  3,
                                     10, 5,  19, 16, 9,  6,  2,  0,  10, 6}},
                {"c_acctbal",
                 Literals<GDF_FLOAT32>{
                     711.56,  121.65,  7498.12, 2866.83, 794.47,  7638.57,
                     9561.95, 6819.74, 8324.07, 2753.54, -272.6,  3396.49,
                     3857.34, 5266.3,  2788.52, 4681.03, 6.34,    5494.43,
                     8914.71, 7603.4,  1428.25, 591.98,  3332.02, 9255.67,
                     7133.7,  5182.05, 5679.84, 1007.18, 7618.27, 9321.01,
                     5236.89, 3471.53, -78.56,  8589.7,  1228.24, 4987.27,
                     -917.75, 6345.11, 6264.31, 1335.3,  270.95,  8727.01,
                     9904.28, 7315.94, 9983.38, 5744.59, 274.58,  3792.5,
                     4573.94, 4266.13}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32",
                Literals<GDF_INT32>{37, 11, 33, 17, 2,  41, 47, 22, 1,  5,
                                    28, 35, 40, 21, 10, 15, 4,  23, 12, 32,
                                    48, 13, 50, 49, 16, 36, 26, 31, 14, 18,
                                    27, 46, 39, 38, 8,  25, 44, 3,  20, 29,
                                    6,  9,  34, 42, 19, 24, 30, 7,  43, 45}},
               {"GDF_INT32",
                Literals<GDF_INT32>{8,  23, 17, 2,  13, 10, 2,  3,  15, 3,
                                    8,  17, 3,  8,  5,  23, 4,  3,  13, 15,
                                    0,  3,  6,  10, 10, 21, 22, 23, 1,  6,
                                    3,  6,  2,  12, 17, 12, 16, 1,  22, 0,
                                    20, 8,  15, 5,  18, 13, 1,  18, 19, 9}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{
                    -917.75, -272.6,  -78.56,  6.34,    121.65,  270.95,
                    274.58,  591.98,  711.56,  794.47,  1007.18, 1228.24,
                    1335.3,  1428.25, 2753.54, 2788.52, 2866.83, 3332.02,
                    3396.49, 3471.53, 3792.5,  3857.34, 4266.13, 4573.94,
                    4681.03, 4987.27, 5182.05, 5236.89, 5266.3,  5494.43,
                    5679.84, 5744.59, 6264.31, 6345.11, 6819.74, 7133.7,
                    7315.94, 7498.12, 7603.4,  7618.27, 7638.57, 8324.07,
                    8589.7,  8727.01, 8914.71, 9255.67, 9321.01, 9561.95,
                    9904.28, 9983.38}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_04) {
  auto input = InputTestItem{
      .query =
          "select c_custkey, c_nationkey, c_acctbal from main.customer order "
          "by c_nationkey, c_acctbal",
      .logicalPlan =
          "LogicalSort(sort0=[$1], sort1=[$2], dir0=[ASC], dir1=[ASC])\n  "
          "LogicalProject(c_custkey=[$0], c_nationkey=[$1], c_acctbal=[$2])\n  "
          "  EnumerableTableScan(table=[[main, customer]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.customer",
               {{"c_custkey",
                 Literals<GDF_INT32>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50}},
                {"c_nationkey",
                 Literals<GDF_INT32>{15, 13, 1,  4,  3,  20, 18, 17, 8,  5,
                                     23, 13, 3,  1,  23, 10, 2,  6,  18, 22,
                                     8,  3,  3,  13, 12, 22, 3,  8,  0,  1,
                                     23, 15, 17, 15, 17, 21, 8,  12, 2,  3,
                                     10, 5,  19, 16, 9,  6,  2,  0,  10, 6}},
                {"c_acctbal",
                 Literals<GDF_FLOAT32>{
                     711.56,  121.65,  7498.12, 2866.83, 794.47,  7638.57,
                     9561.95, 6819.74, 8324.07, 2753.54, -272.6,  3396.49,
                     3857.34, 5266.3,  2788.52, 4681.03, 6.34,    5494.43,
                     8914.71, 7603.4,  1428.25, 591.98,  3332.02, 9255.67,
                     7133.7,  5182.05, 5679.84, 1007.18, 7618.27, 9321.01,
                     5236.89, 3471.53, -78.56,  8589.7,  1228.24, 4987.27,
                     -917.75, 6345.11, 6264.31, 1335.3,  270.95,  8727.01,
                     9904.28, 7315.94, 9983.38, 5744.59, 274.58,  3792.5,
                     4573.94, 4266.13}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32",
                Literals<GDF_INT32>{48, 29, 14, 3,  30, 17, 47, 39, 22, 5,
                                    40, 23, 13, 27, 4,  10, 42, 50, 18, 46,
                                    37, 28, 21, 9,  45, 41, 49, 16, 38, 25,
                                    2,  12, 24, 1,  32, 34, 44, 33, 35, 8,
                                    19, 7,  43, 6,  36, 26, 20, 11, 15, 31}},
               {"GDF_INT32",
                Literals<GDF_INT32>{0,  0,  1,  1,  1,  2,  2,  2,  3,  3,
                                    3,  3,  3,  3,  4,  5,  5,  6,  6,  6,
                                    8,  8,  8,  8,  9,  10, 10, 10, 12, 12,
                                    13, 13, 13, 15, 15, 15, 16, 17, 17, 17,
                                    18, 18, 19, 20, 21, 22, 22, 23, 23, 23}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{
                    3792.5,  7618.27, 5266.3,  7498.12, 9321.01, 6.34,
                    274.58,  6264.31, 591.98,  794.47,  1335.3,  3332.02,
                    3857.34, 5679.84, 2866.83, 2753.54, 8727.01, 4266.13,
                    5494.43, 5744.59, -917.75, 1007.18, 1428.25, 8324.07,
                    9983.38, 270.95,  4573.94, 4681.03, 6345.11, 7133.7,
                    121.65,  3396.49, 9255.67, 711.56,  3471.53, 8589.7,
                    7315.94, -78.56,  1228.24, 6819.74, 8914.71, 9561.95,
                    9904.28, 7638.57, 4987.27, 5182.05, 7603.4,  -272.6,
                    2788.52, 5236.89}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_05) {
  auto input = InputTestItem{
      .query =
          "select c_custkey, c_nationkey, c_acctbal from main.customer order "
          "by c_nationkey, c_custkey",
      .logicalPlan =
          "LogicalSort(sort0=[$1], sort1=[$0], dir0=[ASC], dir1=[ASC])\n  "
          "LogicalProject(c_custkey=[$0], c_nationkey=[$1], c_acctbal=[$2])\n  "
          "  EnumerableTableScan(table=[[main, customer]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.customer",
               {{"c_custkey",
                 Literals<GDF_INT32>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50}},
                {"c_nationkey",
                 Literals<GDF_INT32>{15, 13, 1,  4,  3,  20, 18, 17, 8,  5,
                                     23, 13, 3,  1,  23, 10, 2,  6,  18, 22,
                                     8,  3,  3,  13, 12, 22, 3,  8,  0,  1,
                                     23, 15, 17, 15, 17, 21, 8,  12, 2,  3,
                                     10, 5,  19, 16, 9,  6,  2,  0,  10, 6}},
                {"c_acctbal",
                 Literals<GDF_FLOAT32>{
                     711.56,  121.65,  7498.12, 2866.83, 794.47,  7638.57,
                     9561.95, 6819.74, 8324.07, 2753.54, -272.6,  3396.49,
                     3857.34, 5266.3,  2788.52, 4681.03, 6.34,    5494.43,
                     8914.71, 7603.4,  1428.25, 591.98,  3332.02, 9255.67,
                     7133.7,  5182.05, 5679.84, 1007.18, 7618.27, 9321.01,
                     5236.89, 3471.53, -78.56,  8589.7,  1228.24, 4987.27,
                     -917.75, 6345.11, 6264.31, 1335.3,  270.95,  8727.01,
                     9904.28, 7315.94, 9983.38, 5744.59, 274.58,  3792.5,
                     4573.94, 4266.13}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32",
                Literals<GDF_INT32>{29, 48, 3,  14, 30, 17, 39, 47, 5,  13,
                                    22, 23, 27, 40, 4,  10, 42, 18, 46, 50,
                                    9,  21, 28, 37, 45, 16, 41, 49, 25, 38,
                                    2,  12, 24, 1,  32, 34, 44, 8,  33, 35,
                                    7,  19, 43, 6,  36, 20, 26, 11, 15, 31}},
               {"GDF_INT32",
                Literals<GDF_INT32>{0,  0,  1,  1,  1,  2,  2,  2,  3,  3,
                                    3,  3,  3,  3,  4,  5,  5,  6,  6,  6,
                                    8,  8,  8,  8,  9,  10, 10, 10, 12, 12,
                                    13, 13, 13, 15, 15, 15, 16, 17, 17, 17,
                                    18, 18, 19, 20, 21, 22, 22, 23, 23, 23}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{
                    7618.27, 3792.5,  7498.12, 5266.3,  9321.01, 6.34,
                    6264.31, 274.58,  794.47,  3857.34, 591.98,  3332.02,
                    5679.84, 1335.3,  2866.83, 2753.54, 8727.01, 5494.43,
                    5744.59, 4266.13, 8324.07, 1428.25, 1007.18, -917.75,
                    9983.38, 4681.03, 270.95,  4573.94, 7133.7,  6345.11,
                    121.65,  3396.49, 9255.67, 711.56,  3471.53, 8589.7,
                    7315.94, 6819.74, -78.56,  1228.24, 9561.95, 8914.71,
                    9904.28, 7638.57, 4987.27, 7603.4,  5182.05, -272.6,
                    2788.52, 5236.89}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_06) {
  auto input = InputTestItem{
      .query =
          "select c_custkey + c_nationkey, c_acctbal from main.customer order "
          "by c_custkey",
      .logicalPlan =
          "LogicalProject(EXPR$0=[$0], c_acctbal=[$1])\n  "
          "LogicalSort(sort0=[$2], dir0=[ASC])\n    "
          "LogicalProject(EXPR$0=[+($0, $1)], c_acctbal=[$2], "
          "c_custkey=[$0])\n      EnumerableTableScan(table=[[main, "
          "customer]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.customer",
               {{"c_custkey",
                 Literals<GDF_INT32>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50}},
                {"c_nationkey",
                 Literals<GDF_INT32>{15, 13, 1,  4,  3,  20, 18, 17, 8,  5,
                                     23, 13, 3,  1,  23, 10, 2,  6,  18, 22,
                                     8,  3,  3,  13, 12, 22, 3,  8,  0,  1,
                                     23, 15, 17, 15, 17, 21, 8,  12, 2,  3,
                                     10, 5,  19, 16, 9,  6,  2,  0,  10, 6}},
                {"c_acctbal",
                 Literals<GDF_FLOAT32>{
                     711.56,  121.65,  7498.12, 2866.83, 794.47,  7638.57,
                     9561.95, 6819.74, 8324.07, 2753.54, -272.6,  3396.49,
                     3857.34, 5266.3,  2788.52, 4681.03, 6.34,    5494.43,
                     8914.71, 7603.4,  1428.25, 591.98,  3332.02, 9255.67,
                     7133.7,  5182.05, 5679.84, 1007.18, 7618.27, 9321.01,
                     5236.89, 3471.53, -78.56,  8589.7,  1228.24, 4987.27,
                     -917.75, 6345.11, 6264.31, 1335.3,  270.95,  8727.01,
                     9904.28, 7315.94, 9983.38, 5744.59, 274.58,  3792.5,
                     4573.94, 4266.13}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32",
                Literals<GDF_INT32>{16, 15, 4,  8,  8,  26, 25, 25, 17, 15,
                                    34, 25, 16, 15, 38, 26, 19, 24, 37, 42,
                                    29, 25, 26, 37, 37, 48, 30, 36, 29, 31,
                                    54, 47, 50, 49, 52, 57, 45, 50, 41, 43,
                                    51, 47, 62, 60, 54, 52, 49, 48, 59, 56}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{
                    711.56,  121.65,  7498.12, 2866.83, 794.47,  7638.57,
                    9561.95, 6819.74, 8324.07, 2753.54, -272.6,  3396.49,
                    3857.34, 5266.3,  2788.52, 4681.03, 6.34,    5494.43,
                    8914.71, 7603.4,  1428.25, 591.98,  3332.02, 9255.67,
                    7133.7,  5182.05, 5679.84, 1007.18, 7618.27, 9321.01,
                    5236.89, 3471.53, -78.56,  8589.7,  1228.24, 4987.27,
                    -917.75, 6345.11, 6264.31, 1335.3,  270.95,  8727.01,
                    9904.28, 7315.94, 9983.38, 5744.59, 274.58,  3792.5,
                    4573.94, 4266.13}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_07) {
  auto input = InputTestItem{
      .query =
          "select c_custkey + c_nationkey, c_acctbal, c_custkey - c_nationkey "
          "from main.customer where c_custkey < 25 AND c_nationkey > 5 order "
          "by c_nationkey, c_custkey",
      .logicalPlan =
          "LogicalProject(EXPR$0=[$0], c_acctbal=[$1], EXPR$2=[$2])\n  "
          "LogicalSort(sort0=[$3], sort1=[$4], dir0=[ASC], dir1=[ASC])\n    "
          "LogicalProject(EXPR$0=[+($0, $1)], c_acctbal=[$2], EXPR$2=[-($0, "
          "$1)], c_nationkey=[$1], c_custkey=[$0])\n      "
          "LogicalFilter(condition=[AND(<($0, 25), >($1, 5))])\n        "
          "EnumerableTableScan(table=[[main, customer]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.customer",
               {{"c_custkey",
                 Literals<GDF_INT32>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50}},
                {"c_nationkey",
                 Literals<GDF_INT32>{15, 13, 1,  4,  3,  20, 18, 17, 8,  5,
                                     23, 13, 3,  1,  23, 10, 2,  6,  18, 22,
                                     8,  3,  3,  13, 12, 22, 3,  8,  0,  1,
                                     23, 15, 17, 15, 17, 21, 8,  12, 2,  3,
                                     10, 5,  19, 16, 9,  6,  2,  0,  10, 6}},
                {"c_acctbal",
                 Literals<GDF_FLOAT32>{
                     711.56,  121.65,  7498.12, 2866.83, 794.47,  7638.57,
                     9561.95, 6819.74, 8324.07, 2753.54, -272.6,  3396.49,
                     3857.34, 5266.3,  2788.52, 4681.03, 6.34,    5494.43,
                     8914.71, 7603.4,  1428.25, 591.98,  3332.02, 9255.67,
                     7133.7,  5182.05, 5679.84, 1007.18, 7618.27, 9321.01,
                     5236.89, 3471.53, -78.56,  8589.7,  1228.24, 4987.27,
                     -917.75, 6345.11, 6264.31, 1335.3,  270.95,  8727.01,
                     9904.28, 7315.94, 9983.38, 5744.59, 274.58,  3792.5,
                     4573.94, 4266.13}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32", Literals<GDF_INT32>{24, 17, 29, 26, 15, 25, 37, 16,
                                                 25, 25, 37, 26, 42, 34, 38}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{5494.43, 8324.07, 1428.25, 4681.03,
                                      121.65, 3396.49, 9255.67, 711.56, 6819.74,
                                      9561.95, 8914.71, 7638.57, 7603.4, -272.6,
                                      2788.52}},
               {"GDF_INT32",
                Literals<GDF_INT32>{12, 1, 13, 6, -11, -1, 11, -14, -9, -11, 1,
                                    -14, -2, -12, -8}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_08) {
  auto input = InputTestItem{
      .query =
          "select c_custkey + c_nationkey, c_acctbal from main.customer order "
          "by 1, 2",
      .logicalPlan =
          "LogicalSort(sort0=[$0], sort1=[$1], dir0=[ASC], dir1=[ASC])\n  "
          "LogicalProject(EXPR$0=[+($0, $1)], c_acctbal=[$2])\n    "
          "EnumerableTableScan(table=[[main, customer]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.customer",
               {{"c_custkey",
                 Literals<GDF_INT32>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50}},
                {"c_nationkey",
                 Literals<GDF_INT32>{15, 13, 1,  4,  3,  20, 18, 17, 8,  5,
                                     23, 13, 3,  1,  23, 10, 2,  6,  18, 22,
                                     8,  3,  3,  13, 12, 22, 3,  8,  0,  1,
                                     23, 15, 17, 15, 17, 21, 8,  12, 2,  3,
                                     10, 5,  19, 16, 9,  6,  2,  0,  10, 6}},
                {"c_acctbal",
                 Literals<GDF_FLOAT32>{
                     711.56,  121.65,  7498.12, 2866.83, 794.47,  7638.57,
                     9561.95, 6819.74, 8324.07, 2753.54, -272.6,  3396.49,
                     3857.34, 5266.3,  2788.52, 4681.03, 6.34,    5494.43,
                     8914.71, 7603.4,  1428.25, 591.98,  3332.02, 9255.67,
                     7133.7,  5182.05, 5679.84, 1007.18, 7618.27, 9321.01,
                     5236.89, 3471.53, -78.56,  8589.7,  1228.24, 4987.27,
                     -917.75, 6345.11, 6264.31, 1335.3,  270.95,  8727.01,
                     9904.28, 7315.94, 9983.38, 5744.59, 274.58,  3792.5,
                     4573.94, 4266.13}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32",
                Literals<GDF_INT32>{4,  8,  8,  15, 15, 15, 16, 16, 17, 19,
                                    24, 25, 25, 25, 25, 26, 26, 26, 29, 29,
                                    30, 31, 34, 36, 37, 37, 37, 38, 41, 42,
                                    43, 45, 47, 47, 48, 48, 49, 49, 50, 50,
                                    51, 52, 52, 54, 54, 56, 57, 59, 60, 62}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{
                    7498.12, 794.47,  2866.83, 121.65,  2753.54, 5266.3,
                    711.56,  3857.34, 8324.07, 6.34,    5494.43, 591.98,
                    3396.49, 6819.74, 9561.95, 3332.02, 4681.03, 7638.57,
                    1428.25, 7618.27, 5679.84, 9321.01, -272.6,  1007.18,
                    7133.7,  8914.71, 9255.67, 2788.52, 6264.31, 7603.4,
                    1335.3,  -917.75, 3471.53, 8727.01, 3792.5,  5182.05,
                    274.58,  8589.7,  -78.56,  6345.11, 270.95,  1228.24,
                    5744.59, 5236.89, 9983.38, 4266.13, 4987.27, 4573.94,
                    7315.94, 9904.28}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_09) {
  auto input = InputTestItem{
      .query =
          "select l_quantity, l_orderkey, l_linenumber from main.lineitem "
          "where l_discount > 0.0 and l_quantity < 10.0 and l_orderkey < 500 "
          "order by l_linenumber, l_quantity, l_orderkey",
      .logicalPlan =
          "LogicalSort(sort0=[$2], sort1=[$0], sort2=[$1], dir0=[ASC], "
          "dir1=[ASC], dir2=[ASC])\n  LogicalProject(l_quantity=[$4], "
          "l_orderkey=[$0], l_linenumber=[$3])\n    "
          "LogicalFilter(condition=[AND(>($6, 0.0), <($4, 10.0), <($0, "
          "500))])\n      EnumerableTableScan(table=[[main, lineitem]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.lineitem",
               {{"l_orderkey",
                 Literals<GDF_INT64>{
                     1,   3,   7,   7,   32,  32,  32,  33,  34,  35,  67,  67,
                     68,  69,  70,  70,  71,  99,  103, 129, 131, 162, 163, 165,
                     166, 192, 193, 194, 194, 195, 197, 197, 224, 225, 226, 228,
                     229, 230, 230, 230, 257, 259, 263, 289, 290, 290, 292, 295,
                     322, 322, 323, 325, 326, 327, 353, 354, 356, 385, 387, 389,
                     417, 418, 418, 420, 421, 448, 449, 449, 450, 450, 451, 452,
                     482, 482, 483, 486, 487, 512, 512, 514, 517, 519, 519, 545,
                     547, 548, 551, 576, 576, 576, 576, 579, 579, 582, 583}},
                {"l_partkey",
                 Literals<GDF_INT64>{
                     64,  30,  146, 158, 45,  3,   12,  138, 170, 121, 22,  174,
                     8,   38,  65,  180, 66,  124, 195, 169, 190, 190, 193, 34,
                     46,  197, 93,  184, 57,  85,  178, 106, 51,  172, 118, 5,
                     177, 195, 8,   19,  147, 196, 85,  112, 129, 2,   154, 16,
                     34,  38,  143, 186, 85,  42,  117, 107, 46,  167, 137, 190,
                     132, 2,   35,  101, 134, 170, 109, 10,  107, 79,  87,  115,
                     122, 196, 88,  29,  83,  65,  51,  13,  41,  159, 151, 170,
                     182, 197, 24,  87,  34,  37,  138, 60,  167, 57,  145}},
                {"l_suppkey",
                 Literals<GDF_INT64>{5, 5, 3, 3, 2, 8,  6, 4, 7, 4, 5, 4, 1, 9,
                                     2, 8, 1, 5, 9, 6,  1, 1, 5, 5, 3, 1, 5, 5,
                                     2, 6, 8, 1, 3, 3,  8, 8, 6, 7, 5, 9, 8, 10,
                                     6, 2, 4, 5, 5, 10, 5, 4, 4, 7, 6, 9, 4, 4,
                                     7, 6, 8, 1, 3, 5,  1, 6, 5, 1, 6, 1, 8, 10,
                                     8, 6, 5, 7, 9, 2,  4, 6, 9, 7, 8, 4, 6, 1,
                                     3, 8, 9, 8, 5, 3,  9, 5, 6, 9, 6}},
                {"l_linenumber",
                 Literals<GDF_INT64>{
                     3, 4, 2, 7, 3, 4, 6, 3, 3, 3, 1, 3, 1, 4, 1, 3, 2, 2, 1,
                     7, 3, 1, 4, 1, 4, 4, 1, 2, 5, 1, 2, 6, 6, 1, 5, 1, 4, 2,
                     3, 5, 1, 4, 2, 2, 2, 3, 1, 3, 6, 7, 3, 2, 4, 2, 5, 4, 1,
                     1, 1, 1, 4, 2, 3, 1, 1, 4, 2, 3, 2, 5, 3, 1, 2, 4, 3, 5,
                     2, 5, 7, 3, 3, 1, 6, 1, 3, 1, 1, 1, 2, 3, 4, 3, 6, 1, 1}},
                {"l_quantity",
                 Literals<GDF_FLOAT64>{
                     8.0, 2.0, 9.0, 5.0, 2.0, 4.0, 6.0, 5.0, 6.0, 7.0, 4.0, 5.0,
                     3.0, 3.0, 8.0, 1.0, 3.0, 5.0, 6.0, 1.0, 4.0, 2.0, 5.0, 3.0,
                     8.0, 2.0, 9.0, 1.0, 8.0, 6.0, 8.0, 1.0, 4.0, 4.0, 2.0, 3.0,
                     3.0, 6.0, 1.0, 8.0, 7.0, 3.0, 9.0, 6.0, 2.0, 5.0, 8.0, 8.0,
                     3.0, 5.0, 9.0, 5.0, 5.0, 9.0, 9.0, 7.0, 4.0, 7.0, 1.0, 2.0,
                     2.0, 1.0, 3.0, 5.0, 1.0, 8.0, 4.0, 3.0, 5.0, 2.0, 1.0, 2.0,
                     1.0, 8.0, 9.0, 3.0, 2.0, 6.0, 2.0, 6.0, 9.0, 1.0, 3.0, 4.0,
                     3.0, 2.0, 8.0, 2.0, 6.0, 6.0, 5.0, 6.0, 5.0, 7.0, 1.0}},
                {"l_extendedprice",
                 Literals<GDF_FLOAT64>{
                     7712.48, 1860.06, 9415.26, 5290.75, 1890.08, 3612.0,
                     5472.06, 5190.65, 6421.02, 7147.84, 3688.08, 5370.85,
                     2724.0,  2814.09, 7720.48, 1080.18, 2898.18, 5120.6,
                     6571.14, 1069.16, 4360.76, 2180.38, 5465.95, 2802.09,
                     7568.32, 2194.38, 8937.81, 1084.18, 7656.4,  5910.48,
                     8625.36, 1006.1,  3804.2,  4288.68, 2036.22, 2715.0,
                     3231.51, 6571.14, 908.0,   7352.08, 7329.98, 3288.57,
                     8865.72, 6072.66, 2058.24, 4510.0,  8433.2,  7328.08,
                     2802.09, 4690.15, 9388.26, 5430.9,  4925.4,  8478.36,
                     9153.99, 7049.7,  3784.16, 7470.12, 1037.13, 2180.38,
                     2064.26, 902.0,   2805.09, 5005.5,  1034.13, 8561.36,
                     4036.4,  2730.03, 5035.5,  1958.14, 987.08,  2030.22,
                     1022.12, 8769.52, 8892.72, 2787.06, 1966.16, 5790.36,
                     1902.1,  5478.06, 8469.36, 1059.15, 3153.45, 4280.68,
                     3246.54, 2194.38, 7392.16, 1974.16, 5604.18, 5622.18,
                     5190.65, 5760.36, 5335.8,  6699.35, 1045.14}},
                {"l_discount",
                 Literals<GDF_FLOAT64>{
                     0.1,  0.01, 0.08, 0.04, 0.09, 0.09, 0.04, 0.05, 0.02, 0.06,
                     0.09, 0.03, 0.05, 0.09, 0.03, 0.03, 0.09, 0.02, 0.03, 0.05,
                     0.04, 0.02, 0.02, 0.01, 0.05, 0.06, 0.06, 0.04, 0.04, 0.04,
                     0.09, 0.07, 0.02, 0.09, 0.07, 0.1,  0.02, 0.03, 0.07, 0.09,
                     0.05, 0.08, 0.08, 0.06, 0.05, 0.03, 0.1,  0.1,  0.08, 0.01,
                     0.07, 0.07, 0.03, 0.09, 0.02, 0.06, 0.1,  0.05, 0.08, 0.09,
                     0.01, 0.04, 0.04, 0.04, 0.02, 0.1,  0.1,  0.07, 0.03, 0.09,
                     0.07, 0.04, 0.05, 0.02, 0.04, 0.07, 0.02, 0.03, 0.09, 0.06,
                     0.04, 0.07, 0.04, 0.02, 0.05, 0.06, 0.08, 0.07, 0.06, 0.08,
                     0.03, 0.03, 0.05, 0.07, 0.07}},
                {"l_tax",
                 Literals<GDF_FLOAT64>{
                     0.02, 0.06, 0.08, 0.02, 0.02, 0.03, 0.03, 0.03, 0.06, 0.04,
                     0.04, 0.07, 0.02, 0.04, 0.08, 0.05, 0.07, 0.07, 0.05, 0.04,
                     0.03, 0.01, 0.0,  0.08, 0.02, 0.02, 0.06, 0.06, 0.0,  0.02,
                     0.02, 0.05, 0.0,  0.07, 0.02, 0.08, 0.08, 0.08, 0.06, 0.06,
                     0.02, 0.06, 0.0,  0.05, 0.04, 0.05, 0.03, 0.07, 0.05, 0.02,
                     0.04, 0.08, 0.08, 0.05, 0.02, 0.01, 0.01, 0.06, 0.03, 0.0,
                     0.03, 0.07, 0.06, 0.03, 0.07, 0.0,  0.06, 0.08, 0.02, 0.0,
                     0.05, 0.03, 0.08, 0.05, 0.03, 0.05, 0.06, 0.05, 0.08, 0.01,
                     0.0,  0.07, 0.0,  0.0,  0.02, 0.05, 0.02, 0.01, 0.05, 0.07,
                     0.07, 0.0,  0.08, 0.0,  0.07}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_FLOAT64",
                Literals<GDF_FLOAT64>{
                    1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
                    5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 1.0, 1.0, 1.0,
                    2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 8.0, 9.0,
                    9.0, 9.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0,
                    5.0, 6.0, 7.0, 8.0, 8.0, 9.0, 9.0, 2.0, 2.0, 2.0, 3.0,
                    3.0, 3.0, 4.0, 5.0, 5.0, 7.0, 8.0, 8.0, 8.0, 2.0, 2.0,
                    3.0, 8.0, 8.0, 9.0, 1.0, 3.0, 4.0, 6.0, 1.0, 5.0, 5.0}},
               {"GDF_INT64",
                Literals<GDF_INT64>{
                    387, 421, 162, 389, 452, 68,  165, 228, 67,  225, 356,
                    420, 103, 195, 257, 385, 70,  292, 193, 194, 418, 482,
                    290, 487, 71,  449, 99,  325, 450, 230, 289, 197, 7,
                    263, 327, 70,  230, 451, 32,  418, 449, 131, 33,  67,
                    290, 34,  35,  1,   295, 323, 483, 3,   192, 417, 69,
                    229, 259, 32,  163, 326, 354, 166, 448, 482, 226, 450,
                    486, 194, 230, 353, 197, 322, 224, 32,  129, 7,   322}},
               {"GDF_INT64",
                Literals<GDF_INT64>{
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
TEST_F(EvaluateQueryTest, TEST_10) {
  auto input = InputTestItem{
      .query =
          "select l_quantity, l_orderkey, l_linenumber from main.lineitem "
          "where l_discount > 0.0 and l_quantity < 10.0 and l_orderkey < 500 "
          "order by 3, 1, 2",
      .logicalPlan =
          "LogicalSort(sort0=[$2], sort1=[$0], sort2=[$1], dir0=[ASC], "
          "dir1=[ASC], dir2=[ASC])\n  LogicalProject(l_quantity=[$4], "
          "l_orderkey=[$0], l_linenumber=[$3])\n    "
          "LogicalFilter(condition=[AND(>($6, 0.0), <($4, 10.0), <($0, "
          "500))])\n      EnumerableTableScan(table=[[main, lineitem]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.lineitem",
               {{"l_orderkey",
                 Literals<GDF_INT64>{
                     1,   3,   7,   7,   32,  32,  32,  33,  34,  35,  67,  67,
                     68,  69,  70,  70,  71,  99,  103, 129, 131, 162, 163, 165,
                     166, 192, 193, 194, 194, 195, 197, 197, 224, 225, 226, 228,
                     229, 230, 230, 230, 257, 259, 263, 289, 290, 290, 292, 295,
                     322, 322, 323, 325, 326, 327, 353, 354, 356, 385, 387, 389,
                     417, 418, 418, 420, 421, 448, 449, 449, 450, 450, 451, 452,
                     482, 482, 483, 486, 487, 512, 512, 514, 517, 519, 519, 545,
                     547, 548, 551, 576, 576, 576, 576, 579, 579, 582, 583}},
                {"l_partkey",
                 Literals<GDF_INT64>{
                     64,  30,  146, 158, 45,  3,   12,  138, 170, 121, 22,  174,
                     8,   38,  65,  180, 66,  124, 195, 169, 190, 190, 193, 34,
                     46,  197, 93,  184, 57,  85,  178, 106, 51,  172, 118, 5,
                     177, 195, 8,   19,  147, 196, 85,  112, 129, 2,   154, 16,
                     34,  38,  143, 186, 85,  42,  117, 107, 46,  167, 137, 190,
                     132, 2,   35,  101, 134, 170, 109, 10,  107, 79,  87,  115,
                     122, 196, 88,  29,  83,  65,  51,  13,  41,  159, 151, 170,
                     182, 197, 24,  87,  34,  37,  138, 60,  167, 57,  145}},
                {"l_suppkey",
                 Literals<GDF_INT64>{5, 5, 3, 3, 2, 8,  6, 4, 7, 4, 5, 4, 1, 9,
                                     2, 8, 1, 5, 9, 6,  1, 1, 5, 5, 3, 1, 5, 5,
                                     2, 6, 8, 1, 3, 3,  8, 8, 6, 7, 5, 9, 8, 10,
                                     6, 2, 4, 5, 5, 10, 5, 4, 4, 7, 6, 9, 4, 4,
                                     7, 6, 8, 1, 3, 5,  1, 6, 5, 1, 6, 1, 8, 10,
                                     8, 6, 5, 7, 9, 2,  4, 6, 9, 7, 8, 4, 6, 1,
                                     3, 8, 9, 8, 5, 3,  9, 5, 6, 9, 6}},
                {"l_linenumber",
                 Literals<GDF_INT64>{
                     3, 4, 2, 7, 3, 4, 6, 3, 3, 3, 1, 3, 1, 4, 1, 3, 2, 2, 1,
                     7, 3, 1, 4, 1, 4, 4, 1, 2, 5, 1, 2, 6, 6, 1, 5, 1, 4, 2,
                     3, 5, 1, 4, 2, 2, 2, 3, 1, 3, 6, 7, 3, 2, 4, 2, 5, 4, 1,
                     1, 1, 1, 4, 2, 3, 1, 1, 4, 2, 3, 2, 5, 3, 1, 2, 4, 3, 5,
                     2, 5, 7, 3, 3, 1, 6, 1, 3, 1, 1, 1, 2, 3, 4, 3, 6, 1, 1}},
                {"l_quantity",
                 Literals<GDF_FLOAT64>{
                     8.0, 2.0, 9.0, 5.0, 2.0, 4.0, 6.0, 5.0, 6.0, 7.0, 4.0, 5.0,
                     3.0, 3.0, 8.0, 1.0, 3.0, 5.0, 6.0, 1.0, 4.0, 2.0, 5.0, 3.0,
                     8.0, 2.0, 9.0, 1.0, 8.0, 6.0, 8.0, 1.0, 4.0, 4.0, 2.0, 3.0,
                     3.0, 6.0, 1.0, 8.0, 7.0, 3.0, 9.0, 6.0, 2.0, 5.0, 8.0, 8.0,
                     3.0, 5.0, 9.0, 5.0, 5.0, 9.0, 9.0, 7.0, 4.0, 7.0, 1.0, 2.0,
                     2.0, 1.0, 3.0, 5.0, 1.0, 8.0, 4.0, 3.0, 5.0, 2.0, 1.0, 2.0,
                     1.0, 8.0, 9.0, 3.0, 2.0, 6.0, 2.0, 6.0, 9.0, 1.0, 3.0, 4.0,
                     3.0, 2.0, 8.0, 2.0, 6.0, 6.0, 5.0, 6.0, 5.0, 7.0, 1.0}},
                {"l_extendedprice",
                 Literals<GDF_FLOAT64>{
                     7712.48, 1860.06, 9415.26, 5290.75, 1890.08, 3612.0,
                     5472.06, 5190.65, 6421.02, 7147.84, 3688.08, 5370.85,
                     2724.0,  2814.09, 7720.48, 1080.18, 2898.18, 5120.6,
                     6571.14, 1069.16, 4360.76, 2180.38, 5465.95, 2802.09,
                     7568.32, 2194.38, 8937.81, 1084.18, 7656.4,  5910.48,
                     8625.36, 1006.1,  3804.2,  4288.68, 2036.22, 2715.0,
                     3231.51, 6571.14, 908.0,   7352.08, 7329.98, 3288.57,
                     8865.72, 6072.66, 2058.24, 4510.0,  8433.2,  7328.08,
                     2802.09, 4690.15, 9388.26, 5430.9,  4925.4,  8478.36,
                     9153.99, 7049.7,  3784.16, 7470.12, 1037.13, 2180.38,
                     2064.26, 902.0,   2805.09, 5005.5,  1034.13, 8561.36,
                     4036.4,  2730.03, 5035.5,  1958.14, 987.08,  2030.22,
                     1022.12, 8769.52, 8892.72, 2787.06, 1966.16, 5790.36,
                     1902.1,  5478.06, 8469.36, 1059.15, 3153.45, 4280.68,
                     3246.54, 2194.38, 7392.16, 1974.16, 5604.18, 5622.18,
                     5190.65, 5760.36, 5335.8,  6699.35, 1045.14}},
                {"l_discount",
                 Literals<GDF_FLOAT64>{
                     0.1,  0.01, 0.08, 0.04, 0.09, 0.09, 0.04, 0.05, 0.02, 0.06,
                     0.09, 0.03, 0.05, 0.09, 0.03, 0.03, 0.09, 0.02, 0.03, 0.05,
                     0.04, 0.02, 0.02, 0.01, 0.05, 0.06, 0.06, 0.04, 0.04, 0.04,
                     0.09, 0.07, 0.02, 0.09, 0.07, 0.1,  0.02, 0.03, 0.07, 0.09,
                     0.05, 0.08, 0.08, 0.06, 0.05, 0.03, 0.1,  0.1,  0.08, 0.01,
                     0.07, 0.07, 0.03, 0.09, 0.02, 0.06, 0.1,  0.05, 0.08, 0.09,
                     0.01, 0.04, 0.04, 0.04, 0.02, 0.1,  0.1,  0.07, 0.03, 0.09,
                     0.07, 0.04, 0.05, 0.02, 0.04, 0.07, 0.02, 0.03, 0.09, 0.06,
                     0.04, 0.07, 0.04, 0.02, 0.05, 0.06, 0.08, 0.07, 0.06, 0.08,
                     0.03, 0.03, 0.05, 0.07, 0.07}},
                {"l_tax",
                 Literals<GDF_FLOAT64>{
                     0.02, 0.06, 0.08, 0.02, 0.02, 0.03, 0.03, 0.03, 0.06, 0.04,
                     0.04, 0.07, 0.02, 0.04, 0.08, 0.05, 0.07, 0.07, 0.05, 0.04,
                     0.03, 0.01, 0.0,  0.08, 0.02, 0.02, 0.06, 0.06, 0.0,  0.02,
                     0.02, 0.05, 0.0,  0.07, 0.02, 0.08, 0.08, 0.08, 0.06, 0.06,
                     0.02, 0.06, 0.0,  0.05, 0.04, 0.05, 0.03, 0.07, 0.05, 0.02,
                     0.04, 0.08, 0.08, 0.05, 0.02, 0.01, 0.01, 0.06, 0.03, 0.0,
                     0.03, 0.07, 0.06, 0.03, 0.07, 0.0,  0.06, 0.08, 0.02, 0.0,
                     0.05, 0.03, 0.08, 0.05, 0.03, 0.05, 0.06, 0.05, 0.08, 0.01,
                     0.0,  0.07, 0.0,  0.0,  0.02, 0.05, 0.02, 0.01, 0.05, 0.07,
                     0.07, 0.0,  0.08, 0.0,  0.07}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_FLOAT64",
                Literals<GDF_FLOAT64>{
                    1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
                    5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 1.0, 1.0, 1.0,
                    2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 8.0, 9.0,
                    9.0, 9.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0,
                    5.0, 6.0, 7.0, 8.0, 8.0, 9.0, 9.0, 2.0, 2.0, 2.0, 3.0,
                    3.0, 3.0, 4.0, 5.0, 5.0, 7.0, 8.0, 8.0, 8.0, 2.0, 2.0,
                    3.0, 8.0, 8.0, 9.0, 1.0, 3.0, 4.0, 6.0, 1.0, 5.0, 5.0}},
               {"GDF_INT64",
                Literals<GDF_INT64>{
                    387, 421, 162, 389, 452, 68,  165, 228, 67,  225, 356,
                    420, 103, 195, 257, 385, 70,  292, 193, 194, 418, 482,
                    290, 487, 71,  449, 99,  325, 450, 230, 289, 197, 7,
                    263, 327, 70,  230, 451, 32,  418, 449, 131, 33,  67,
                    290, 34,  35,  1,   295, 323, 483, 3,   192, 417, 69,
                    229, 259, 32,  163, 326, 354, 166, 448, 482, 226, 450,
                    486, 194, 230, 353, 197, 322, 224, 32,  129, 7,   322}},
               {"GDF_INT64",
                Literals<GDF_INT64>{
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
