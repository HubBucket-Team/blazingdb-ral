
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
TEST_F(EvaluateQueryTest, TEST_01) {
  auto input = InputTestItem{
      .query = "select * from main.emps",
      .logicalPlan =
          "LogicalProject(id=[$0], age=[$1])\n  "
          "EnumerableTableScan(table=[[main, emps]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.emps",
               {{"id", Literals<GDF_INT8>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1}},
                {"age",
                 Literals<GDF_INT8>{10, 20, 30, 40, 50, 60, 70, 80, 90, 10}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT8", Literals<GDF_INT8>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1}},
               {"GDF_INT8",
                Literals<GDF_INT8>{10, 20, 30, 40, 50, 60, 70, 80, 90, 10}}}}
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
      .query = "select id > 3 from main.emps",
      .logicalPlan =
          "LogicalProject(EXPR$0=[>($0, 3)])\n  "
          "EnumerableTableScan(table=[[main, emps]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.emps",
               {{"id", Literals<GDF_INT8>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1}},
                {"age",
                 Literals<GDF_INT8>{10, 20, 30, 40, 50, 60, 70, 80, 90, 10}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT8", Literals<GDF_INT8>{0, 0, 0, 1, 1, 1, 1, 1, 1, 0}}}}
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
      .query = "select id from main.emps where age > 30",
      .logicalPlan =
          "LogicalProject(id=[$0])\n  LogicalFilter(condition=[>($1, 30)])\n   "
          " EnumerableTableScan(table=[[main, emps]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.emps",
               {{"id", Literals<GDF_INT8>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1}},
                {"age",
                 Literals<GDF_INT8>{10, 20, 30, 40, 50, 60, 70, 80, 90, 10}}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT8", Literals<GDF_INT8>{4, 5, 6, 7, 8, 9, 0, 0, 0, 0}}}} //Todo: check the zeroes at the final output, (hardcoding output)
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
      .query = "select age + salary from main.emps",
      .logicalPlan =
          "LogicalProject(EXPR$0=[+($1, $2)])\n  "
          "EnumerableTableScan(table=[[main, emps]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.emps",
               {{"id", Literals<GDF_INT8>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1}},
                {"age",
                 Literals<GDF_INT8>{10, 20, 30, 40, 50, 60, 70, 80, 90, 10}},
                {"salary",
                 Literals<GDF_INT8>{90, 80, 70, 60, 50, 40, 30, 20, 10, 0
                 }}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT8", Literals<GDF_INT8>{100, 100, 100, 100, 100, 100,
                                               100, 100, 100, 10}}}}
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
      .query = "select salary from main.emps where age > 80",
      .logicalPlan =
          "LogicalProject(salary=[$2])\n  LogicalFilter(condition=[>($1, "
          "80)])\n    EnumerableTableScan(table=[[main, emps]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.emps",
               {{"id", Literals<GDF_INT8>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1}},
                {"age",
                 Literals<GDF_INT8>{10, 20, 30, 40, 50, 60, 70, 80, 90, 10}},
                {"salary",
                 Literals<GDF_INT8>{
                     90, 80, 70, 60, 50, 40, 30, 20, 10, 0
                 }}}}}
              .Build(),
      .resultTable = LiteralTableBuilder{"ResultSet",
                                         {{"GDF_INT8",
                                           Literals<GDF_INT8>{
                                               10, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                           }}}} //Hardcoding output, zeroes at the end
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
      .query = "select * from main.emps where age = 10",
      .logicalPlan =
          "LogicalProject(id=[$0], age=[$1], salary=[$2])\n  "
          "LogicalFilter(condition=[=($1, 10)])\n    "
          "EnumerableTableScan(table=[[main, emps]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.emps",
               {{"id", Literals<GDF_INT8>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1}},
                {"age",
                 Literals<GDF_INT8>{10, 20, 10, 20, 10, 20, 10, 20, 10, 2}},
                {"salary",
                 Literals<GDF_INT8>{
                     90, 80, 70, 60, 50, 40, 30, 20, 10, 0
                 }}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT8",
                Literals<GDF_INT8>{
                    1, 3, 5, 7, 9, 0, 0, 0, 0, 0
                }},
               {"GDF_INT8", Literals<GDF_INT8>{10, 10, 10, 10, 10, 0, 0, 0, 0, 0}},
               {"GDF_INT8", Literals<GDF_INT8>{90, 70, 50, 30, 10, 0, 0, 0, 0, 0}}}}
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
      .query = "select * from main.emps where age = 10 and salary > 4999",
      .logicalPlan =
          "LogicalProject(id=[$0], age=[$1], salary=[$2])\n  "
          "LogicalFilter(condition=[AND(=($1, 10), >($2, 4999))])\n    "
          "EnumerableTableScan(table=[[main, emps]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.emps",
               {{"id", Literals<GDF_INT32>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1}},
                {"age",
                 Literals<GDF_INT32>{10, 20, 10, 20, 10, 20, 10, 20, 10, 2}},
                {"salary",
                 Literals<GDF_INT32>{
                     9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 0
                 }}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32",
                Literals<GDF_INT32>{
                    1, 3, 5, 0, 0, 0, 0, 0, 0, 0
                }},
               {"GDF_INT32", Literals<GDF_INT32>{10, 10, 10, 0, 0, 0, 0, 0, 0, 0}},
               {"GDF_INT32", Literals<GDF_INT32>{9000, 7000, 5000, 0, 0, 0, 0, 0, 0, 0}}}}
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
      .query = "select id + salary from main.emps",
      .logicalPlan =
          "LogicalProject(EXPR$0=[+($0, $2)])\n  "
          "EnumerableTableScan(table=[[main, emps]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.emps",
               {{"id", Literals<GDF_INT32>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1}},
                {"age",
                 Literals<GDF_INT32>{10, 20, 10, 20, 10, 20, 10, 20, 10, 2}},
                {"salary",
                 Literals<GDF_INT32>{
                     9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 0
                 }}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{
              "ResultSet",
              {{"GDF_INT32", Literals<GDF_INT32>{9001, 8002, 7003, 6004, 5005,
                                                 4006, 3007, 2008, 1009, 1}}}}
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
      .query = "select age * salary from main.emps where id < 5 and age = 10",
      .logicalPlan =
          "LogicalProject(EXPR$0=[*(10, $2)])\n  "
          "LogicalFilter(condition=[AND(<($0, 5), =($1, 10))])\n    "
          "EnumerableTableScan(table=[[main, emps]])",
      .tableGroup =
          LiteralTableGroupBuilder{
              {"main.emps",
               {{"id", Literals<GDF_INT32>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1}},
                {"age",
                 Literals<GDF_INT32>{10, 20, 10, 20, 10, 20, 10, 20, 10, 2}},
                {"salary",
                 Literals<GDF_INT32>{
                     9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000,
                 }}}}}
              .Build(),
      .resultTable =
          LiteralTableBuilder{"ResultSet",
                              {{"GDF_INT32", Literals<GDF_INT32>{90000, 70000, 0, 0, 0, 0, 0, 0, 0, 0}}}}
              .Build()};
  auto logical_plan = input.logicalPlan;
  auto input_tables = input.tableGroup.ToBlazingFrame();
  auto table_names = input.tableGroup.table_names();
  auto column_names = input.tableGroup.column_names();
  std::vector<gdf_column_cpp> outputs;
  gdf_error err = evaluate_query(input_tables, table_names, column_names,
                                 logical_plan, outputs);
  std::cout<<"is this valid bitmask null? "<<(input_tables[0][0].get_gdf_column()->valid == nullptr)<<std::endl;
  print_gdf_column( input_tables[0][0].get_gdf_column());
  print_gdf_column( input_tables[0][1].get_gdf_column());

  print_gdf_column(  outputs[0].get_gdf_column());
std::cout<<"bummer"<<std::endl;

  EXPECT_TRUE(err == GDF_SUCCESS);
  auto output_table =
      GdfColumnCppsTableBuilder{"output_table", outputs}.Build();
  CHECK_RESULT(output_table, input.resultTable);
}
