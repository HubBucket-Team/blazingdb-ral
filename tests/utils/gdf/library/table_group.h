#pragma once
#include "column.h"
#include "table.h"

namespace gdf {
namespace library {

using BlazingFrame = std::vector<std::vector<gdf_column_cpp> >;

class TableGroup {
public:
  TableGroup(std::initializer_list<Table> tables) {
    for (Table table : tables) { tables_.push_back(table); }
  }

  TableGroup(const std::vector<Table> &tables) : tables_{tables} {}

  BlazingFrame ToBlazingFrame() const;

  const Table &operator[](const std::size_t i) const { return tables_[i]; }

private:
  std::vector<Table> tables_;
};

BlazingFrame TableGroup::ToBlazingFrame() const {
  BlazingFrame frame;
  frame.resize(tables_.size());
  std::transform(tables_.cbegin(),
                 tables_.cend(),
                 frame.begin(),
                 [](const Table &table) { return table.ToGdfColumnCpps(); });
  return frame;
}

class TableGroupBuilder {
public:
  TableGroupBuilder(std::initializer_list<TableBuilder> builders)
    : builders_{builders} {}

  TableGroup Build(const std::initializer_list<const std::size_t> lengths) {
    std::vector<Table> tables;
    tables.resize(builders_.size());
    std::transform(std::begin(builders_),
                   std::end(builders_),
                   tables.begin(),
                   [this, lengths](const TableBuilder &builder) {
                     return builder.Build(
                       *(std::begin(lengths)
                         + std::distance(std::begin(builders_), &builder)));
                   });
    return TableGroup(tables);
  }

  TableGroup Build(const std::size_t length) {
    std::vector<Table> tables;
    tables.resize(builders_.size());
    std::transform(
      std::begin(builders_),
      std::end(builders_),
      tables.begin(),
      [length](const TableBuilder &builder) { return builder.Build(length); });
    return TableGroup{tables};
  }

private:
  std::initializer_list<TableBuilder> builders_;
};

using Index = const std::size_t;

}  // namespace library
}  // namespace gdf
