
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <CalciteExpressionParsing.h>
#include <CalciteInterpreter.h>
#include <DataFrame.h>
#include <StringUtil.h>
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
      .query =
          "select n.n_nationkey, r.r_regionkey from main.nation as n left "
          "outer join main.region as r on n.n_nationkey = r.r_regionkey where "
          "n.n_nationkey < 10",
      .logicalPlan =
          "LogicalProject(n_nationkey=[$0], r_regionkey=[$4])\n  "
          "LogicalFilter(condition=[<($0, 10)])\n    "
          "LogicalProject(n_nationkey=[$0], n_name=[$1], n_regionkey=[$2], "
          "n_comment=[$3], r_regionkey=[$5], r_name=[$6], r_comment=[$7])\n    "
          "  LogicalJoin(condition=[=($4, $5)], joinType=[left])\n        "
          "LogicalProject(n_nationkey=[$0], n_name=[$1], n_regionkey=[$2], "
          "n_comment=[$3], n_nationkey0=[CAST($0):BIGINT NOT NULL])\n          "
          "EnumerableTableScan(table=[[main, nation]])\n        "
          "EnumerableTableScan(table=[[main, region]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.nation",
               {{"n_nationkey",
                 Literals<GDF_INT32>{0,  1,  2,  3,  4,  5,  6,  7,  8,
                                     9,  10, 11, 12, 13, 14, 15, 16, 17,
                                     18, 19, 20, 21, 22, 23, 24}},
                {"n_name",
                 Literals<GDF_INT64>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                {"n_regionkey",
                 Literals<GDF_INT32>{0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2,
                                     4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1}},
                {"n_comment",
                 Literals<GDF_INT64>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}}},
              {"main.region",
               {{"r_regionkey", Literals<GDF_INT64>{0, 0, 0, 0, 0}},
                {"r_name", Literals<GDF_INT64>{0, 0, 0, 0, 0}},
                {"r_comment", Literals<GDF_INT64>{0, 0, 0, 0, 0}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT64", Literals<GDF_INT64>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
               {"GDF_FLOAT64",
                Literals<GDF_FLOAT64>{0, 1, 2, 3, 4, -1, -1, -1, -1, -1}}}}
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
      .query =
          "select n.n_nationkey, r.r_regionkey, n.n_nationkey + r.r_regionkey "
          "from main.nation as n left outer join main.region as r on "
          "n.n_nationkey = r.r_regionkey where n.n_nationkey < 10",
      .logicalPlan =
          "LogicalProject(n_nationkey=[$0], r_regionkey=[$4], EXPR$2=[+($0, "
          "$4)])\n  LogicalFilter(condition=[<($0, 10)])\n    "
          "LogicalProject(n_nationkey=[$0], n_name=[$1], n_regionkey=[$2], "
          "n_comment=[$3], r_regionkey=[$5], r_name=[$6], r_comment=[$7])\n    "
          "  LogicalJoin(condition=[=($4, $5)], joinType=[left])\n        "
          "LogicalProject(n_nationkey=[$0], n_name=[$1], n_regionkey=[$2], "
          "n_comment=[$3], n_nationkey0=[CAST($0):BIGINT NOT NULL])\n          "
          "EnumerableTableScan(table=[[main, nation]])\n        "
          "EnumerableTableScan(table=[[main, region]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.nation",
               {{"n_nationkey",
                 Literals<GDF_INT32>{0,  1,  2,  3,  4,  5,  6,  7,  8,
                                     9,  10, 11, 12, 13, 14, 15, 16, 17,
                                     18, 19, 20, 21, 22, 23, 24}},
                {"n_name",
                 Literals<GDF_INT64>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                {"n_regionkey",
                 Literals<GDF_INT32>{0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2,
                                     4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1}},
                {"n_comment",
                 Literals<GDF_INT64>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}}},
              {"main.region",
               {{"r_regionkey", Literals<GDF_INT64>{0, 0, 0, 0, 0}},
                {"r_name", Literals<GDF_INT64>{0, 0, 0, 0, 0}},
                {"r_comment", Literals<GDF_INT64>{0, 0, 0, 0, 0}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT64", Literals<GDF_INT64>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
               {"GDF_FLOAT64",
                Literals<GDF_FLOAT64>{0, 1, 2, 3, 4, -1, -1, -1, -1, -1}},
               {"GDF_FLOAT64",
                Literals<GDF_FLOAT64>{0, 2, 4, 6, 8, -1, -1, -1, -1, -1}}}}
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
          "select n.n_nationkey, r.r_regionkey from main.nation as n left "
          "outer join main.region as r on n.n_regionkey = r.r_regionkey where "
          "n.n_nationkey < 10",
      .logicalPlan =
          "LogicalProject(n_nationkey=[$0], r_regionkey=[$4])\n  "
          "LogicalFilter(condition=[<($0, 10)])\n    "
          "LogicalProject(n_nationkey=[$0], n_name=[$1], n_regionkey=[$2], "
          "n_comment=[$3], r_regionkey=[$5], r_name=[$6], r_comment=[$7])\n    "
          "  LogicalJoin(condition=[=($4, $5)], joinType=[left])\n        "
          "LogicalProject(n_nationkey=[$0], n_name=[$1], n_regionkey=[$2], "
          "n_comment=[$3], n_regionkey0=[CAST($2):BIGINT NOT NULL])\n          "
          "EnumerableTableScan(table=[[main, nation]])\n        "
          "EnumerableTableScan(table=[[main, region]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.nation",
               {{"n_nationkey",
                 Literals<GDF_INT32>{0,  1,  2,  3,  4,  5,  6,  7,  8,
                                     9,  10, 11, 12, 13, 14, 15, 16, 17,
                                     18, 19, 20, 21, 22, 23, 24}},
                {"n_name",
                 Literals<GDF_INT64>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                {"n_regionkey",
                 Literals<GDF_INT32>{0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2,
                                     4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1}},
                {"n_comment",
                 Literals<GDF_INT64>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}}},
              {"main.region",
               {{"r_regionkey", Literals<GDF_INT64>{0, 0, 0, 0, 0}},
                {"r_name", Literals<GDF_INT64>{0, 0, 0, 0, 0}},
                {"r_comment", Literals<GDF_INT64>{0, 0, 0, 0, 0}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT64", Literals<GDF_INT64>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
               {"GDF_INT64",
                Literals<GDF_INT64>{0, 1, 1, 1, 4, 0, 3, 3, 2, 2}}}}
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
          "select n.n_nationkey, r.r_regionkey from main.nation as n left "
          "outer join main.region as r on n.n_regionkey = r.r_regionkey where "
          "n.n_nationkey < 10 and n.n_nationkey > 5",
      .logicalPlan =
          "LogicalProject(n_nationkey=[$0], r_regionkey=[$4])\n  "
          "LogicalFilter(condition=[AND(<($0, 10), >($0, 5))])\n    "
          "LogicalProject(n_nationkey=[$0], n_name=[$1], n_regionkey=[$2], "
          "n_comment=[$3], r_regionkey=[$5], r_name=[$6], r_comment=[$7])\n    "
          "  LogicalJoin(condition=[=($4, $5)], joinType=[left])\n        "
          "LogicalProject(n_nationkey=[$0], n_name=[$1], n_regionkey=[$2], "
          "n_comment=[$3], n_regionkey0=[CAST($2):BIGINT NOT NULL])\n          "
          "EnumerableTableScan(table=[[main, nation]])\n        "
          "EnumerableTableScan(table=[[main, region]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.nation",
               {{"n_nationkey",
                 Literals<GDF_INT32>{0,  1,  2,  3,  4,  5,  6,  7,  8,
                                     9,  10, 11, 12, 13, 14, 15, 16, 17,
                                     18, 19, 20, 21, 22, 23, 24}},
                {"n_name",
                 Literals<GDF_INT64>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                {"n_regionkey",
                 Literals<GDF_INT32>{0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2,
                                     4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1}},
                {"n_comment",
                 Literals<GDF_INT64>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}}},
              {"main.region",
               {{"r_regionkey", Literals<GDF_INT64>{0, 0, 0, 0, 0}},
                {"r_name", Literals<GDF_INT64>{0, 0, 0, 0, 0}},
                {"r_comment", Literals<GDF_INT64>{0, 0, 0, 0, 0}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{"ResultSet",
                              {{"GDF_INT64", Literals<GDF_INT64>{6, 7, 8, 9}},
                               {"GDF_INT64", Literals<GDF_INT64>{3, 3, 2, 2}}}}
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
