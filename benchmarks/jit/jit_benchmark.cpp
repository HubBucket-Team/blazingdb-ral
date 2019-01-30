/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo
 * <cristhian@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <CalciteInterpreter.h>
#include <DataFrame.h>
#include <benchmark/benchmark.h>
#include <GDFColumn.cuh>
#include <Utils.cuh>
#include <algorithm>
#include <cstdlib>
#include <utility>
#include "Interpreter/interpreter_cpp.h"
#include "gdf/library/api.h"

using namespace gdf::library;

template <int logPlan>
struct JitBench : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    logicalPlan = "LogicalProject(EXPR$0=[+($0, $2)])";
    logicalPlan = "LogicalProject(EXPR$0=[*(*(10, SIN(/(+($0, $2), $1))), POWER(/(-($0, $2), $1), 2))], EXPR$1=[-(FLOOR(+(POWER(SIN($1), 2), POWER(COS($1), 2))), CEIL(+(POWER(SIN(+($1, 53.42)), 2), POWER(COS(+($1, 53.42)), 2))))], EXPR$2=[+(-(/(*(FLOOR(+($1, 0.1)), POWER(MOD($0, 13), 2.0)), 5), CEIL(MOD(*(2, $2), 57))), 0.001)])";
  }

  void TearDown(benchmark::State& state) override {}

  std::string logicalPlan;
};

// namespace {
// char const* LOGICAL_PLANS[] = {"LogicalProject(EXPR$0=[+($0, $2)])"};
// }

// template <char const* logPlan>
// struct BenchParameters {
//   static constexpr char const* logicalPlan{logPlan};
// };

BENCHMARK_TEMPLATE_DEFINE_F(JitBench, SimpleBench, 4)
(benchmark::State& state) {
  std::vector<int32_t> x;
  std::vector<double> y;
  std::vector<int32_t> z;

  x.resize(state.range(0));
  std::generate(x.begin(), x.end(),
                []() { return std::rand() % (RAND_MAX / 13); });
  y.resize(state.range(0));
  std::generate(y.begin(), y.end(), []() { return ((std::rand() % RAND_MAX) + 1); });
  z.resize(state.range(0));
  std::generate(z.begin(), z.end(),
                []() { return std::rand() % (RAND_MAX / 2); });

  gdf::library::TableGroup tableGroup =
      LiteralTableGroupBuilder{{"temp",
                                {{"in_x", Literals<GDF_INT32>{std::move(x)}},
                                 {"in_y", Literals<GDF_FLOAT64>{std::move(y)}},
                                 {"in_z", Literals<GDF_INT32>{std::move(z)}}}}}
          .Build();
  auto input_tables = tableGroup.ToBlazingFrame();

  blazing_frame bz_frame;

  gdf_error err = GDF_SUCCESS;
  for (auto _ : state) {
    state.PauseTiming();
    bz_frame.clear();
    for (auto& t : input_tables) bz_frame.add_table(t);
    state.ResumeTiming();

    err = process_project(bz_frame, logicalPlan);
  }
}

BENCHMARK_REGISTER_F(JitBench, SimpleBench)
    ->Range(50 << 10, 50 << 20)
    ->Unit(benchmark::kMillisecond);
