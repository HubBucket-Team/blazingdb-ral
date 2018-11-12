
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
                    4,   8,   8,   15,  15,  15,  16,  16,  17,  19,  24,  25,
                    25,  25,  25,  26,  26,  26,  29,  29,  30,  31,  34,  36,
                    37,  37,  37,  38,  41,  42,  43,  45,  47,  47,  48,  48,
                    49,  49,  50,  50,  51,  52,  52,  54,  54,  56,  57,  58,
                    59,  60,  60,  62,  63,  63,  65,  66,  67,  68,  69,  71,
                    72,  73,  74,  76,  76,  78,  78,  78,  78,  78,  80,  80,
                    84,  86,  87,  88,  88,  90,  92,  93,  94,  94,  94,  95,
                    99,  100, 100, 101, 103, 103, 103, 104, 104, 105, 106, 107,
                    110, 110, 110, 112, 113, 114, 114, 114, 115, 120, 120, 121,
                    122, 123, 125, 125, 125, 126, 128, 128, 131, 132, 132, 132,
                    133, 136, 136, 136, 138, 139, 141, 142, 142, 142, 143, 143,
                    144, 144, 145, 145, 148, 148, 148, 149, 150, 151, 153, 154,
                    158, 159, 159, 165, 168, 168}},
               {"GDF_FLOAT32",
                Literals<GDF_FLOAT32>{
                    7498.12, 794.47,  2866.83, 121.65,  2753.54, 5266.3,
                    711.56,  3857.34, 8324.07, 6.34,    5494.43, 591.98,
                    3396.49, 6819.74, 9561.95, 3332.02, 4681.03, 7638.57,
                    1428.25, 7618.27, 5679.84, 9321.01, -272.6,  1007.18,
                    7133.7,  8914.71, 9255.67, 2788.52, 6264.31, 7603.4,
                    1335.3,  -917.75, 3471.53, 8727.01, 3792.5,  5182.05,
                    274.58,  8589.7,  -78.56,  6345.11, 270.95,  1228.24,
                    5744.59, 5236.89, 9983.38, 4266.13, 4987.27, 868.9,
                    4573.94, 3458.6,  7315.94, 9904.28, 855.87,  5630.28,
                    4572.11, 6530.86, -646.64, 4113.64, 595.61,  6478.46,
                    2741.87, 4288.5,  -362.86, 5745.33, 8166.59, -611.19,
                    1536.24, 1709.28, 2764.43, 4151.93, 6853.37, 7383.53,
                    9331.13, 3306.32, 7136.97, 242.77,  8795.16, 3386.64,
                    4867.52, 6684.1,  1182.91, 1738.87, 5121.28, 5174.71,
                    4643.14, 2182.52, 9468.34, 2023.71, 1530.76, 5500.11,
                    7470.96, 6323.92, 8031.44, 6463.51, 7354.23, 3288.42,
                    -551.37, 5327.38, 6327.54, 2757.45, 2259.38, -588.38,
                    2164.48, 4088.65, 9091.82, 7462.99, 9889.89, 8462.17,
                    2514.15, 7508.92, -716.1,  2912.0,  7865.46, 3930.35,
                    1027.46, 5897.83, 2953.35, -986.96, 363.75,  8403.99,
                    6505.26, 162.57,  3582.37, 9127.27, 6428.32, 5073.58,
                    3950.83, 1842.49, 6706.14, 8595.53, -842.39, 430.59,
                    -234.12, 9963.15, 4608.9,  6417.31, 1001.39, 7897.78,
                    9280.71, 3328.68, 2314.67, 2209.81, 7838.3,  8732.91,
                    9748.93, 2135.6,  2186.5,  8071.4,  3849.48, 8959.65}}}}
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
                     1,   1,   1,   1,   1,   1,   2,   3,   3,   3,   3,   3,
                     3,   4,   5,   5,   5,   6,   7,   7,   7,   7,   7,   7,
                     7,   32,  32,  32,  32,  32,  32,  33,  33,  33,  33,  34,
                     34,  34,  35,  35,  35,  35,  35,  35,  36,  37,  37,  37,
                     38,  39,  39,  39,  39,  39,  39,  64,  65,  65,  65,  66,
                     66,  67,  67,  67,  67,  67,  67,  68,  68,  68,  68,  68,
                     68,  68,  69,  69,  69,  69,  69,  69,  70,  70,  70,  70,
                     70,  70,  71,  71,  71,  71,  71,  71,  96,  96,  97,  97,
                     97,  98,  98,  98,  98,  99,  99,  99,  99,  100, 100, 100,
                     100, 100, 101, 101, 101, 102, 102, 102, 102, 103, 103, 103,
                     103, 128, 129, 129, 129, 129, 129, 129, 129, 130, 130, 130,
                     130, 130, 131, 131, 131, 132, 132, 132, 132, 133, 133, 133,
                     133, 134, 134, 134, 134, 134, 134, 135, 135, 135, 135, 135,
                     135, 160, 160, 160, 161, 162, 163, 163, 163, 163, 163, 163,
                     164, 164, 164, 164, 164, 164, 164, 165, 165, 165, 165, 165,
                     166, 166, 166, 166, 167, 167, 192, 192, 192, 192, 192, 192,
                     193, 193, 193, 194, 194, 194, 194, 194}},
                {"l_partkey",
                 Literals<GDF_INT64>{
                     156, 68,  64,  3,   25,  16,  107, 5,   20,  129, 30,  184,
                     63,  89,  109, 124, 38,  140, 183, 146, 95,  164, 152, 80,
                     158, 83,  198, 45,  3,   86,  12,  62,  61,  138, 34,  89,
                     90,  170, 1,   162, 121, 86,  120, 31,  120, 23,  127, 13,
                     176, 3,   187, 68,  21,  55,  95,  86,  60,  74,  2,   116,
                     174, 22,  21,  174, 88,  41,  179, 8,   176, 35,  95,  83,
                     103, 140, 116, 105, 138, 38,  93,  19,  65,  197, 180, 46,
                     38,  56,  62,  66,  35,  97,  104, 196, 124, 136, 120, 50,
                     78,  41,  110, 45,  168, 88,  124, 135, 109, 63,  116, 47,
                     39,  54,  119, 164, 139, 89,  170, 183, 62,  195, 11,  29,
                     30,  107, 3,   186, 40,  136, 32,  78,  169, 129, 2,   12,
                     116, 70,  168, 45,  190, 141, 120, 115, 29,  104, 177, 118,
                     90,  1,   165, 189, 145, 36,  134, 109, 199, 158, 68,  137,
                     115, 15,  87,  21,  103, 190, 168, 121, 37,  193, 127, 191,
                     92,  19,  126, 18,  148, 109, 4,   34,  162, 59,  140, 156,
                     65,  167, 100, 46,  102, 172, 98,  162, 111, 197, 83,  142,
                     93,  154, 94,  3,   184, 66,  146, 57}},
                {"l_suppkey",
                 Literals<GDF_INT64>{
                     4,  9,  5,  6,  8,  3,  2,  2, 10, 8,  5,  5,  8,  10, 10,
                     5,  4,  6,  4,  3,  8,  5,  4, 10, 3,  4,  10, 2,  8,  7,
                     6,  7,  8,  4,  5,  10, 1,  7, 4,  1,  4,  7,  7,  7,  1,
                     8,  6,  7,  5,  10, 8,  3,  6, 10, 7,  7,  5,  3,  5,  10,
                     5,  5,  10, 4,  9,  10, 9,  1, 4,  1,  9,  4,  6,  6,  10,
                     10, 4,  9,  6,  3,  2,  10, 8, 9,  9,  8,  3,  1,  1,  9,
                     7,  9,  7,  7,  4,  7,  6,  2, 7,  6,  9,  9,  5,  1,  2,
                     4,  10, 4,  10, 6,  9,  9,  5, 10, 5,  4,  7,  9,  5,  10,
                     9,  10, 6,  7,  6,  7,  8,  6, 6,  10, 5,  3,  6,  7,  7,
                     8,  1,  8,  1,  6,  2,  7,  5, 8,  1,  2,  2,  10, 6,  7,
                     10, 10, 3,  10, 7,  8,  5,  2, 8,  10, 10, 1,  3,  2,  3,
                     5,  2,  4,  4,  6,  9,  2,  1, 10, 7,  5,  7,  1,  1,  4,
                     2,  8,  2,  3,  3,  2,  1,  7, 8,  1,  4,  9,  5,  6,  6,
                     6,  5,  1,  7,  2}},
                {"l_linenumber",
                 Literals<GDF_INT64>{
                     1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 1, 1,
                     2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 1, 2, 3,
                     1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 1, 1, 2, 3, 4, 5, 6, 1, 1,
                     2, 3, 1, 2, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 1, 2,
                     3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 1,
                     2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 1,
                     2, 3, 4, 1, 2, 3, 4, 1, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4,
                     5, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 1,
                     2, 3, 4, 5, 6, 1, 2, 3, 1, 1, 1, 2, 3, 4, 5, 6, 1, 2, 3,
                     4, 5, 6, 7, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4,
                     5, 6, 1, 2, 3, 1, 2, 3, 4, 5}},
                {"l_quantity",
                 Literals<GDF_FLOAT64>{
                     17.0, 36.0, 8.0,  28.0, 24.0, 32.0, 38.0, 45.0, 49.0,
                     27.0, 2.0,  28.0, 26.0, 30.0, 15.0, 26.0, 50.0, 37.0,
                     12.0, 9.0,  46.0, 28.0, 38.0, 35.0, 5.0,  28.0, 32.0,
                     2.0,  4.0,  44.0, 6.0,  31.0, 32.0, 5.0,  41.0, 13.0,
                     22.0, 6.0,  24.0, 34.0, 7.0,  25.0, 34.0, 28.0, 42.0,
                     40.0, 39.0, 43.0, 44.0, 44.0, 26.0, 46.0, 32.0, 43.0,
                     40.0, 21.0, 26.0, 22.0, 21.0, 31.0, 41.0, 4.0,  12.0,
                     5.0,  44.0, 23.0, 29.0, 3.0,  46.0, 46.0, 20.0, 27.0,
                     30.0, 41.0, 48.0, 32.0, 17.0, 3.0,  42.0, 23.0, 8.0,
                     13.0, 1.0,  11.0, 37.0, 19.0, 25.0, 3.0,  45.0, 33.0,
                     39.0, 34.0, 23.0, 30.0, 13.0, 37.0, 19.0, 28.0, 1.0,
                     14.0, 10.0, 10.0, 5.0,  42.0, 36.0, 28.0, 22.0, 46.0,
                     14.0, 37.0, 49.0, 36.0, 12.0, 37.0, 34.0, 25.0, 15.0,
                     6.0,  37.0, 23.0, 32.0, 38.0, 46.0, 36.0, 33.0, 34.0,
                     24.0, 22.0, 1.0,  14.0, 48.0, 18.0, 13.0, 31.0, 45.0,
                     50.0, 4.0,  18.0, 43.0, 32.0, 23.0, 27.0, 12.0, 29.0,
                     11.0, 21.0, 35.0, 26.0, 47.0, 12.0, 12.0, 47.0, 21.0,
                     33.0, 34.0, 20.0, 13.0, 36.0, 22.0, 34.0, 19.0, 2.0,
                     43.0, 13.0, 27.0, 5.0,  12.0, 20.0, 26.0, 24.0, 38.0,
                     32.0, 43.0, 27.0, 23.0, 3.0,  43.0, 15.0, 49.0, 27.0,
                     37.0, 13.0, 41.0, 8.0,  28.0, 27.0, 23.0, 20.0, 15.0,
                     2.0,  25.0, 45.0, 9.0,  15.0, 23.0, 17.0, 1.0,  13.0,
                     36.0, 8.0}},
                {"l_extendedprice",
                 Literals<GDF_FLOAT64>{
                     17954.55, 34850.16, 7712.48,  25284.0,  22200.48, 29312.32,
                     38269.8,  40725.0,  45080.98, 27786.24, 1860.06,  30357.04,
                     25039.56, 29672.4,  15136.5,  26627.12, 46901.5,  38485.18,
                     12998.16, 9415.26,  45774.14, 29796.48, 39981.7,  34302.8,
                     5290.75,  27526.24, 35142.08, 1890.08,  3612.0,   43387.52,
                     5472.06,  29823.86, 30753.92, 5190.65,  38295.23, 12858.04,
                     21781.98, 6421.02,  21624.0,  36113.44, 7147.84,  24652.0,
                     34684.08, 26068.84, 42845.04, 36920.8,  40057.68, 39259.43,
                     47351.48, 39732.0,  28266.68, 44530.76, 29472.64, 41067.15,
                     39803.6,  20707.68, 24961.56, 21429.54, 18942.0,  31499.41,
                     44040.97, 3688.08,  11052.24, 5370.85,  43475.52, 21643.92,
                     31295.93, 2724.0,   49503.82, 43011.38, 19901.8,  26543.16,
                     30093.0,  42645.74, 48773.28, 32163.2,  17648.21, 2814.09,
                     41709.78, 21137.23, 7720.48,  14263.47, 1080.18,  10406.44,
                     34707.11, 18164.95, 24051.5,  2898.18,  42076.35, 32903.97,
                     39159.9,  37270.46, 23554.76, 31083.9,  13261.56, 35151.85,
                     18583.33, 26349.12, 1010.11,  13230.56, 10681.6,  9880.8,
                     5120.6,   43475.46, 36327.6,  26965.68, 22354.42, 43563.84,
                     13146.42, 35299.85, 49936.39, 38309.76, 12469.56, 36595.96,
                     36385.78, 27079.5,  14430.9,  6571.14,  33707.37, 21367.46,
                     29760.96, 38269.8,  41538.0,  39102.48, 31021.32, 35228.42,
                     22368.72, 21517.54, 1069.16,  14407.68, 43296.0,  16416.18,
                     13209.43, 30072.17, 48067.2,  47252.0,  4360.76,  18740.52,
                     43865.16, 32483.52, 21367.46, 27110.7,  12926.04, 29525.19,
                     10890.99, 18921.0,  37280.6,  28318.68, 49121.58, 11232.36,
                     12409.56, 47427.7,  23082.99, 34918.95, 32914.04, 20742.6,
                     13196.43, 32940.36, 21715.76, 31314.68, 19058.9,  2180.38,
                     45930.88, 13274.56, 25299.81, 5465.95,  12325.44, 21823.8,
                     25794.34, 22056.24, 38992.56, 29376.32, 45070.02, 27245.7,
                     20792.0,  2802.09,  45672.88, 14385.75, 50966.86, 28516.05,
                     35707.22, 13873.08, 41004.1,  7568.32,  28058.8,  28948.59,
                     22956.07, 21243.2,  15166.65, 2194.38,  24577.0,  46896.3,
                     8937.81,  15812.25, 22864.07, 15351.0,  1084.18,  12558.78,
                     37661.04, 7656.4}},
                {"l_discount",
                 Literals<GDF_FLOAT64>{
                     0.04, 0.09, 0.1,  0.09, 0.1,  0.07, 0.0,  0.06, 0.1,
                     0.06, 0.01, 0.04, 0.1,  0.03, 0.02, 0.07, 0.08, 0.08,
                     0.07, 0.08, 0.1,  0.03, 0.08, 0.06, 0.04, 0.05, 0.02,
                     0.09, 0.09, 0.05, 0.04, 0.09, 0.02, 0.05, 0.09, 0.0,
                     0.08, 0.02, 0.02, 0.06, 0.06, 0.06, 0.08, 0.03, 0.09,
                     0.09, 0.05, 0.05, 0.04, 0.09, 0.08, 0.06, 0.07, 0.01,
                     0.06, 0.05, 0.03, 0.0,  0.09, 0.0,  0.04, 0.09, 0.09,
                     0.03, 0.08, 0.05, 0.02, 0.05, 0.02, 0.04, 0.07, 0.03,
                     0.05, 0.09, 0.01, 0.08, 0.09, 0.09, 0.07, 0.05, 0.03,
                     0.06, 0.03, 0.01, 0.09, 0.06, 0.09, 0.09, 0.0,  0.0,
                     0.08, 0.04, 0.1,  0.01, 0.0,  0.02, 0.06, 0.06, 0.0,
                     0.05, 0.03, 0.02, 0.02, 0.02, 0.09, 0.04, 0.0,  0.03,
                     0.06, 0.05, 0.1,  0.0,  0.06, 0.06, 0.03, 0.01, 0.07,
                     0.03, 0.02, 0.01, 0.01, 0.06, 0.08, 0.01, 0.04, 0.0,
                     0.06, 0.06, 0.05, 0.08, 0.03, 0.04, 0.09, 0.06, 0.1,
                     0.02, 0.04, 0.0,  0.01, 0.04, 0.1,  0.0,  0.02, 0.09,
                     0.06, 0.0,  0.06, 0.09, 0.05, 0.05, 0.0,  0.06, 0.0,
                     0.02, 0.02, 0.01, 0.04, 0.07, 0.0,  0.01, 0.01, 0.02,
                     0.01, 0.01, 0.04, 0.02, 0.1,  0.0,  0.09, 0.05, 0.03,
                     0.05, 0.06, 0.1,  0.09, 0.01, 0.08, 0.0,  0.07, 0.01,
                     0.09, 0.09, 0.07, 0.05, 0.06, 0.09, 0.0,  0.07, 0.09,
                     0.06, 0.02, 0.0,  0.06, 0.02, 0.06, 0.05, 0.04, 0.08,
                     0.0,  0.04}},
                {"l_tax",
                 Literals<GDF_FLOAT64>{
                     0.02, 0.06, 0.02, 0.06, 0.04, 0.02, 0.05, 0.0,  0.0,
                     0.07, 0.06, 0.0,  0.02, 0.08, 0.04, 0.08, 0.03, 0.03,
                     0.03, 0.08, 0.07, 0.04, 0.01, 0.03, 0.02, 0.08, 0.0,
                     0.02, 0.03, 0.06, 0.03, 0.04, 0.05, 0.03, 0.0,  0.07,
                     0.06, 0.06, 0.0,  0.08, 0.04, 0.05, 0.06, 0.02, 0.0,
                     0.03, 0.02, 0.08, 0.02, 0.06, 0.04, 0.08, 0.05, 0.01,
                     0.05, 0.02, 0.03, 0.05, 0.07, 0.08, 0.07, 0.04, 0.05,
                     0.07, 0.06, 0.07, 0.05, 0.02, 0.05, 0.05, 0.01, 0.06,
                     0.06, 0.08, 0.07, 0.06, 0.0,  0.04, 0.04, 0.0,  0.08,
                     0.06, 0.05, 0.05, 0.04, 0.03, 0.07, 0.07, 0.07, 0.01,
                     0.06, 0.01, 0.06, 0.06, 0.02, 0.06, 0.08, 0.07, 0.0,
                     0.02, 0.03, 0.01, 0.07, 0.02, 0.02, 0.05, 0.07, 0.04,
                     0.03, 0.0,  0.0,  0.01, 0.02, 0.0,  0.08, 0.01, 0.07,
                     0.05, 0.07, 0.04, 0.07, 0.01, 0.02, 0.02, 0.06, 0.01,
                     0.0,  0.01, 0.04, 0.05, 0.02, 0.08, 0.02, 0.05, 0.02,
                     0.04, 0.03, 0.08, 0.08, 0.04, 0.0,  0.02, 0.06, 0.08,
                     0.01, 0.03, 0.07, 0.06, 0.0,  0.02, 0.0,  0.08, 0.07,
                     0.0,  0.03, 0.04, 0.02, 0.01, 0.04, 0.05, 0.01, 0.01,
                     0.0,  0.04, 0.08, 0.0,  0.0,  0.07, 0.04, 0.05, 0.06,
                     0.01, 0.01, 0.04, 0.04, 0.08, 0.05, 0.05, 0.06, 0.04,
                     0.03, 0.05, 0.03, 0.02, 0.01, 0.0,  0.0,  0.01, 0.01,
                     0.02, 0.03, 0.05, 0.06, 0.07, 0.05, 0.04, 0.06, 0.08,
                     0.05, 0.0}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_FLOAT64",
                Literals<GDF_FLOAT64>{2.0, 3.0, 3.0, 4.0, 6.0, 8.0, 9.0, 1.0,
                                      3.0, 5.0, 9.0, 1.0, 2.0, 4.0, 5.0, 5.0,
                                      6.0, 7.0, 8.0, 2.0, 2.0, 3.0, 4.0, 5.0,
                                      8.0, 8.0, 6.0, 1.0, 5.0}},
               {"GDF_INT64",
                Literals<GDF_INT64>{162, 68,  165, 67,  103, 70,  193, 194,
                                    71,  99,  7,   70,  32,  131, 33,  67,
                                    34,  35,  1,   3,   192, 69,  32,  163,
                                    166, 194, 32,  129, 7}},
               {"GDF_INT64",
                Literals<GDF_INT64>{1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                    3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 6, 7, 7}}}}
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
                     1,   1,   1,   1,   1,   1,   2,   3,   3,   3,   3,   3,
                     3,   4,   5,   5,   5,   6,   7,   7,   7,   7,   7,   7,
                     7,   32,  32,  32,  32,  32,  32,  33,  33,  33,  33,  34,
                     34,  34,  35,  35,  35,  35,  35,  35,  36,  37,  37,  37,
                     38,  39,  39,  39,  39,  39,  39,  64,  65,  65,  65,  66,
                     66,  67,  67,  67,  67,  67,  67,  68,  68,  68,  68,  68,
                     68,  68,  69,  69,  69,  69,  69,  69,  70,  70,  70,  70,
                     70,  70,  71,  71,  71,  71,  71,  71,  96,  96,  97,  97,
                     97,  98,  98,  98,  98,  99,  99,  99,  99,  100, 100, 100,
                     100, 100, 101, 101, 101, 102, 102, 102, 102, 103, 103, 103,
                     103, 128, 129, 129, 129, 129, 129, 129, 129, 130, 130, 130,
                     130, 130, 131, 131, 131, 132, 132, 132, 132, 133, 133, 133,
                     133, 134, 134, 134, 134, 134, 134, 135, 135, 135, 135, 135,
                     135, 160, 160, 160, 161, 162, 163, 163, 163, 163, 163, 163,
                     164, 164, 164, 164, 164, 164, 164, 165, 165, 165, 165, 165,
                     166, 166, 166, 166, 167, 167, 192, 192, 192, 192, 192, 192,
                     193, 193, 193, 194, 194, 194, 194, 194}},
                {"l_partkey",
                 Literals<GDF_INT64>{
                     156, 68,  64,  3,   25,  16,  107, 5,   20,  129, 30,  184,
                     63,  89,  109, 124, 38,  140, 183, 146, 95,  164, 152, 80,
                     158, 83,  198, 45,  3,   86,  12,  62,  61,  138, 34,  89,
                     90,  170, 1,   162, 121, 86,  120, 31,  120, 23,  127, 13,
                     176, 3,   187, 68,  21,  55,  95,  86,  60,  74,  2,   116,
                     174, 22,  21,  174, 88,  41,  179, 8,   176, 35,  95,  83,
                     103, 140, 116, 105, 138, 38,  93,  19,  65,  197, 180, 46,
                     38,  56,  62,  66,  35,  97,  104, 196, 124, 136, 120, 50,
                     78,  41,  110, 45,  168, 88,  124, 135, 109, 63,  116, 47,
                     39,  54,  119, 164, 139, 89,  170, 183, 62,  195, 11,  29,
                     30,  107, 3,   186, 40,  136, 32,  78,  169, 129, 2,   12,
                     116, 70,  168, 45,  190, 141, 120, 115, 29,  104, 177, 118,
                     90,  1,   165, 189, 145, 36,  134, 109, 199, 158, 68,  137,
                     115, 15,  87,  21,  103, 190, 168, 121, 37,  193, 127, 191,
                     92,  19,  126, 18,  148, 109, 4,   34,  162, 59,  140, 156,
                     65,  167, 100, 46,  102, 172, 98,  162, 111, 197, 83,  142,
                     93,  154, 94,  3,   184, 66,  146, 57}},
                {"l_suppkey",
                 Literals<GDF_INT64>{
                     4,  9,  5,  6,  8,  3,  2,  2, 10, 8,  5,  5,  8,  10, 10,
                     5,  4,  6,  4,  3,  8,  5,  4, 10, 3,  4,  10, 2,  8,  7,
                     6,  7,  8,  4,  5,  10, 1,  7, 4,  1,  4,  7,  7,  7,  1,
                     8,  6,  7,  5,  10, 8,  3,  6, 10, 7,  7,  5,  3,  5,  10,
                     5,  5,  10, 4,  9,  10, 9,  1, 4,  1,  9,  4,  6,  6,  10,
                     10, 4,  9,  6,  3,  2,  10, 8, 9,  9,  8,  3,  1,  1,  9,
                     7,  9,  7,  7,  4,  7,  6,  2, 7,  6,  9,  9,  5,  1,  2,
                     4,  10, 4,  10, 6,  9,  9,  5, 10, 5,  4,  7,  9,  5,  10,
                     9,  10, 6,  7,  6,  7,  8,  6, 6,  10, 5,  3,  6,  7,  7,
                     8,  1,  8,  1,  6,  2,  7,  5, 8,  1,  2,  2,  10, 6,  7,
                     10, 10, 3,  10, 7,  8,  5,  2, 8,  10, 10, 1,  3,  2,  3,
                     5,  2,  4,  4,  6,  9,  2,  1, 10, 7,  5,  7,  1,  1,  4,
                     2,  8,  2,  3,  3,  2,  1,  7, 8,  1,  4,  9,  5,  6,  6,
                     6,  5,  1,  7,  2}},
                {"l_linenumber",
                 Literals<GDF_INT64>{
                     1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 1, 1,
                     2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 1, 2, 3,
                     1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 1, 1, 2, 3, 4, 5, 6, 1, 1,
                     2, 3, 1, 2, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 1, 2,
                     3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 1,
                     2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 1,
                     2, 3, 4, 1, 2, 3, 4, 1, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4,
                     5, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 1,
                     2, 3, 4, 5, 6, 1, 2, 3, 1, 1, 1, 2, 3, 4, 5, 6, 1, 2, 3,
                     4, 5, 6, 7, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4,
                     5, 6, 1, 2, 3, 1, 2, 3, 4, 5}},
                {"l_quantity",
                 Literals<GDF_FLOAT64>{
                     17.0, 36.0, 8.0,  28.0, 24.0, 32.0, 38.0, 45.0, 49.0,
                     27.0, 2.0,  28.0, 26.0, 30.0, 15.0, 26.0, 50.0, 37.0,
                     12.0, 9.0,  46.0, 28.0, 38.0, 35.0, 5.0,  28.0, 32.0,
                     2.0,  4.0,  44.0, 6.0,  31.0, 32.0, 5.0,  41.0, 13.0,
                     22.0, 6.0,  24.0, 34.0, 7.0,  25.0, 34.0, 28.0, 42.0,
                     40.0, 39.0, 43.0, 44.0, 44.0, 26.0, 46.0, 32.0, 43.0,
                     40.0, 21.0, 26.0, 22.0, 21.0, 31.0, 41.0, 4.0,  12.0,
                     5.0,  44.0, 23.0, 29.0, 3.0,  46.0, 46.0, 20.0, 27.0,
                     30.0, 41.0, 48.0, 32.0, 17.0, 3.0,  42.0, 23.0, 8.0,
                     13.0, 1.0,  11.0, 37.0, 19.0, 25.0, 3.0,  45.0, 33.0,
                     39.0, 34.0, 23.0, 30.0, 13.0, 37.0, 19.0, 28.0, 1.0,
                     14.0, 10.0, 10.0, 5.0,  42.0, 36.0, 28.0, 22.0, 46.0,
                     14.0, 37.0, 49.0, 36.0, 12.0, 37.0, 34.0, 25.0, 15.0,
                     6.0,  37.0, 23.0, 32.0, 38.0, 46.0, 36.0, 33.0, 34.0,
                     24.0, 22.0, 1.0,  14.0, 48.0, 18.0, 13.0, 31.0, 45.0,
                     50.0, 4.0,  18.0, 43.0, 32.0, 23.0, 27.0, 12.0, 29.0,
                     11.0, 21.0, 35.0, 26.0, 47.0, 12.0, 12.0, 47.0, 21.0,
                     33.0, 34.0, 20.0, 13.0, 36.0, 22.0, 34.0, 19.0, 2.0,
                     43.0, 13.0, 27.0, 5.0,  12.0, 20.0, 26.0, 24.0, 38.0,
                     32.0, 43.0, 27.0, 23.0, 3.0,  43.0, 15.0, 49.0, 27.0,
                     37.0, 13.0, 41.0, 8.0,  28.0, 27.0, 23.0, 20.0, 15.0,
                     2.0,  25.0, 45.0, 9.0,  15.0, 23.0, 17.0, 1.0,  13.0,
                     36.0, 8.0}},
                {"l_extendedprice",
                 Literals<GDF_FLOAT64>{
                     17954.55, 34850.16, 7712.48,  25284.0,  22200.48, 29312.32,
                     38269.8,  40725.0,  45080.98, 27786.24, 1860.06,  30357.04,
                     25039.56, 29672.4,  15136.5,  26627.12, 46901.5,  38485.18,
                     12998.16, 9415.26,  45774.14, 29796.48, 39981.7,  34302.8,
                     5290.75,  27526.24, 35142.08, 1890.08,  3612.0,   43387.52,
                     5472.06,  29823.86, 30753.92, 5190.65,  38295.23, 12858.04,
                     21781.98, 6421.02,  21624.0,  36113.44, 7147.84,  24652.0,
                     34684.08, 26068.84, 42845.04, 36920.8,  40057.68, 39259.43,
                     47351.48, 39732.0,  28266.68, 44530.76, 29472.64, 41067.15,
                     39803.6,  20707.68, 24961.56, 21429.54, 18942.0,  31499.41,
                     44040.97, 3688.08,  11052.24, 5370.85,  43475.52, 21643.92,
                     31295.93, 2724.0,   49503.82, 43011.38, 19901.8,  26543.16,
                     30093.0,  42645.74, 48773.28, 32163.2,  17648.21, 2814.09,
                     41709.78, 21137.23, 7720.48,  14263.47, 1080.18,  10406.44,
                     34707.11, 18164.95, 24051.5,  2898.18,  42076.35, 32903.97,
                     39159.9,  37270.46, 23554.76, 31083.9,  13261.56, 35151.85,
                     18583.33, 26349.12, 1010.11,  13230.56, 10681.6,  9880.8,
                     5120.6,   43475.46, 36327.6,  26965.68, 22354.42, 43563.84,
                     13146.42, 35299.85, 49936.39, 38309.76, 12469.56, 36595.96,
                     36385.78, 27079.5,  14430.9,  6571.14,  33707.37, 21367.46,
                     29760.96, 38269.8,  41538.0,  39102.48, 31021.32, 35228.42,
                     22368.72, 21517.54, 1069.16,  14407.68, 43296.0,  16416.18,
                     13209.43, 30072.17, 48067.2,  47252.0,  4360.76,  18740.52,
                     43865.16, 32483.52, 21367.46, 27110.7,  12926.04, 29525.19,
                     10890.99, 18921.0,  37280.6,  28318.68, 49121.58, 11232.36,
                     12409.56, 47427.7,  23082.99, 34918.95, 32914.04, 20742.6,
                     13196.43, 32940.36, 21715.76, 31314.68, 19058.9,  2180.38,
                     45930.88, 13274.56, 25299.81, 5465.95,  12325.44, 21823.8,
                     25794.34, 22056.24, 38992.56, 29376.32, 45070.02, 27245.7,
                     20792.0,  2802.09,  45672.88, 14385.75, 50966.86, 28516.05,
                     35707.22, 13873.08, 41004.1,  7568.32,  28058.8,  28948.59,
                     22956.07, 21243.2,  15166.65, 2194.38,  24577.0,  46896.3,
                     8937.81,  15812.25, 22864.07, 15351.0,  1084.18,  12558.78,
                     37661.04, 7656.4}},
                {"l_discount",
                 Literals<GDF_FLOAT64>{
                     0.04, 0.09, 0.1,  0.09, 0.1,  0.07, 0.0,  0.06, 0.1,
                     0.06, 0.01, 0.04, 0.1,  0.03, 0.02, 0.07, 0.08, 0.08,
                     0.07, 0.08, 0.1,  0.03, 0.08, 0.06, 0.04, 0.05, 0.02,
                     0.09, 0.09, 0.05, 0.04, 0.09, 0.02, 0.05, 0.09, 0.0,
                     0.08, 0.02, 0.02, 0.06, 0.06, 0.06, 0.08, 0.03, 0.09,
                     0.09, 0.05, 0.05, 0.04, 0.09, 0.08, 0.06, 0.07, 0.01,
                     0.06, 0.05, 0.03, 0.0,  0.09, 0.0,  0.04, 0.09, 0.09,
                     0.03, 0.08, 0.05, 0.02, 0.05, 0.02, 0.04, 0.07, 0.03,
                     0.05, 0.09, 0.01, 0.08, 0.09, 0.09, 0.07, 0.05, 0.03,
                     0.06, 0.03, 0.01, 0.09, 0.06, 0.09, 0.09, 0.0,  0.0,
                     0.08, 0.04, 0.1,  0.01, 0.0,  0.02, 0.06, 0.06, 0.0,
                     0.05, 0.03, 0.02, 0.02, 0.02, 0.09, 0.04, 0.0,  0.03,
                     0.06, 0.05, 0.1,  0.0,  0.06, 0.06, 0.03, 0.01, 0.07,
                     0.03, 0.02, 0.01, 0.01, 0.06, 0.08, 0.01, 0.04, 0.0,
                     0.06, 0.06, 0.05, 0.08, 0.03, 0.04, 0.09, 0.06, 0.1,
                     0.02, 0.04, 0.0,  0.01, 0.04, 0.1,  0.0,  0.02, 0.09,
                     0.06, 0.0,  0.06, 0.09, 0.05, 0.05, 0.0,  0.06, 0.0,
                     0.02, 0.02, 0.01, 0.04, 0.07, 0.0,  0.01, 0.01, 0.02,
                     0.01, 0.01, 0.04, 0.02, 0.1,  0.0,  0.09, 0.05, 0.03,
                     0.05, 0.06, 0.1,  0.09, 0.01, 0.08, 0.0,  0.07, 0.01,
                     0.09, 0.09, 0.07, 0.05, 0.06, 0.09, 0.0,  0.07, 0.09,
                     0.06, 0.02, 0.0,  0.06, 0.02, 0.06, 0.05, 0.04, 0.08,
                     0.0,  0.04}},
                {"l_tax",
                 Literals<GDF_FLOAT64>{
                     0.02, 0.06, 0.02, 0.06, 0.04, 0.02, 0.05, 0.0,  0.0,
                     0.07, 0.06, 0.0,  0.02, 0.08, 0.04, 0.08, 0.03, 0.03,
                     0.03, 0.08, 0.07, 0.04, 0.01, 0.03, 0.02, 0.08, 0.0,
                     0.02, 0.03, 0.06, 0.03, 0.04, 0.05, 0.03, 0.0,  0.07,
                     0.06, 0.06, 0.0,  0.08, 0.04, 0.05, 0.06, 0.02, 0.0,
                     0.03, 0.02, 0.08, 0.02, 0.06, 0.04, 0.08, 0.05, 0.01,
                     0.05, 0.02, 0.03, 0.05, 0.07, 0.08, 0.07, 0.04, 0.05,
                     0.07, 0.06, 0.07, 0.05, 0.02, 0.05, 0.05, 0.01, 0.06,
                     0.06, 0.08, 0.07, 0.06, 0.0,  0.04, 0.04, 0.0,  0.08,
                     0.06, 0.05, 0.05, 0.04, 0.03, 0.07, 0.07, 0.07, 0.01,
                     0.06, 0.01, 0.06, 0.06, 0.02, 0.06, 0.08, 0.07, 0.0,
                     0.02, 0.03, 0.01, 0.07, 0.02, 0.02, 0.05, 0.07, 0.04,
                     0.03, 0.0,  0.0,  0.01, 0.02, 0.0,  0.08, 0.01, 0.07,
                     0.05, 0.07, 0.04, 0.07, 0.01, 0.02, 0.02, 0.06, 0.01,
                     0.0,  0.01, 0.04, 0.05, 0.02, 0.08, 0.02, 0.05, 0.02,
                     0.04, 0.03, 0.08, 0.08, 0.04, 0.0,  0.02, 0.06, 0.08,
                     0.01, 0.03, 0.07, 0.06, 0.0,  0.02, 0.0,  0.08, 0.07,
                     0.0,  0.03, 0.04, 0.02, 0.01, 0.04, 0.05, 0.01, 0.01,
                     0.0,  0.04, 0.08, 0.0,  0.0,  0.07, 0.04, 0.05, 0.06,
                     0.01, 0.01, 0.04, 0.04, 0.08, 0.05, 0.05, 0.06, 0.04,
                     0.03, 0.05, 0.03, 0.02, 0.01, 0.0,  0.0,  0.01, 0.01,
                     0.02, 0.03, 0.05, 0.06, 0.07, 0.05, 0.04, 0.06, 0.08,
                     0.05, 0.0}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_FLOAT64",
                Literals<GDF_FLOAT64>{2.0, 3.0, 3.0, 4.0, 6.0, 8.0, 9.0, 1.0,
                                      3.0, 5.0, 9.0, 1.0, 2.0, 4.0, 5.0, 5.0,
                                      6.0, 7.0, 8.0, 2.0, 2.0, 3.0, 4.0, 5.0,
                                      8.0, 8.0, 6.0, 1.0, 5.0}},
               {"GDF_INT64",
                Literals<GDF_INT64>{162, 68,  165, 67,  103, 70,  193, 194,
                                    71,  99,  7,   70,  32,  131, 33,  67,
                                    34,  35,  1,   3,   192, 69,  32,  163,
                                    166, 194, 32,  129, 7}},
               {"GDF_INT64",
                Literals<GDF_INT64>{1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                    3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 6, 7, 7}}}}
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
