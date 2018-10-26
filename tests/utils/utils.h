#ifndef RAL_TEST_UTILS_H_
#define RAL_TEST_UTILS_H_

#include <algorithm>
#include <cassert>
#include <functional>
#include <string>
#include <vector>

#include <GDFColumn.cuh>
#include <gdf/gdf.h>

namespace ral {
namespace test {
namespace utils {

template <gdf_dtype DTYPE>
struct DTypeTraits {};

#define DTYPE_FACTORY(DTYPE, T)                                                \
  template <>                                                                  \
  struct DTypeTraits<GDF_##DTYPE> {                                            \
    typedef T value_type;                                                      \
  }

DTYPE_FACTORY(INT8, std::int8_t);
DTYPE_FACTORY(INT16, std::int16_t);
DTYPE_FACTORY(INT32, std::int32_t);
DTYPE_FACTORY(INT64, std::int64_t);
DTYPE_FACTORY(UINT8, std::uint8_t);
DTYPE_FACTORY(UINT16, std::uint16_t);
DTYPE_FACTORY(UINT32, std::uint32_t);
DTYPE_FACTORY(UINT64, std::uint64_t);
DTYPE_FACTORY(FLOAT32, float);
DTYPE_FACTORY(FLOAT64, double);
DTYPE_FACTORY(DATE32, std::int32_t);
DTYPE_FACTORY(DATE64, std::int64_t);
DTYPE_FACTORY(TIMESTAMP, std::int64_t);

#undef DTYPE_FACTORY

class Column {
public:
  virtual ~Column();

  virtual gdf_column_cpp ToGdfColumnCpp() const         = 0;
  virtual const void *   get(const std::size_t i) const = 0;

  class Wrapper {
  public:
    template <class T>
    operator T() const {
      return *static_cast<const T *>(column->get(i));
    }

    template <gdf_dtype DType>
    typename DTypeTraits<DType>::value_type get() const {
      return static_cast<typename DTypeTraits<DType>::value_type>(*this);
    }

    const std::size_t i;
    const Column *    column;
  };

  template <gdf_dtype DType>
  typename DTypeTraits<DType>::value_type
  get(const std::size_t i) const {  //! \deprecated
    return Wrapper{i, this}.get<DType>();
  }

  Wrapper operator[](const std::size_t i) const { return Wrapper{i, this}; }

protected:
  static gdf_column_cpp Create(const gdf_dtype   dtype,
                               const std::size_t length,
                               const void *      data,
                               const std::size_t size);
};

template <gdf_dtype GDF_DType>
class DType {
public:
  using value_type = typename DTypeTraits<GDF_DType>::value_type;

  static constexpr gdf_dtype value  = GDF_DType;
  static constexpr std::size_t size = sizeof(value_type);

  template <class T>
  DType(const T value) : value_{value} {}

  operator value_type() const { return value_; }

private:
  const value_type value_;
};

template <gdf_dtype value>
using Ret = DType<value>;  //! \deprecated

template <gdf_dtype dType>
class TypedColumn : public Column {
public:
  using value_type = typename DTypeTraits<dType>::value_type;
  using Callback   = std::function<value_type(const std::size_t)>;

  TypedColumn(const std::string &name) : name_{name} {}

  void
  range(const std::size_t begin, const std::size_t end, Callback callback) {
    assert(end > begin);
    values_.reserve(end - begin);
    for (std::size_t i = begin; i < end; i++) {
      values_.push_back(callback(i));
    }
  }

  void range(const std::size_t end, Callback callback) {
    range(0, end, callback);
  }

  gdf_column_cpp ToGdfColumnCpp() const final {
    using DT = DType<dType>;
    return Create(DT::value, values_.length(), values_.data(), DT::size);
  }

  value_type  operator[](const std::size_t i) const { return values_[i]; }
  const void *get(const std::size_t i) const final { return &values_[i]; }

private:
  const std::string name_;

  std::basic_string<value_type> values_;
};

class Table {
public:
  Table(const std::string &                     name,
        std::vector<std::shared_ptr<Column> > &&columns)
    : name_{name}, columns_{std::move(columns)} {}

  Table() {}                              //! \deprecated
  Table &operator=(const Table &other) {  //! \deprecated
    name_        = other.name_;
    columns_     = other.columns_;
    return *this;
  }

  const Column &operator[](const std::size_t i) const { return *columns_[i]; }

  std::vector<gdf_column_cpp> ToGdfColumnCpps() const;

private:
  std::string name_;

  std::vector<std::shared_ptr<Column> > columns_;
};

using BlazingFrame = std::vector<std::vector<gdf_column_cpp> >;

class TableGroup {
public:
  TableGroup(std::initializer_list<Table> tables) {
    for (Table table : tables) { tables_.push_back(table); }
  }

  TableGroup(const std::vector<Table> &tables) {
    for (Table table : tables) { tables_.push_back(table); }
  }

  BlazingFrame ToBlazingFrame() const;

  const Table &operator[](const std::size_t i) const { return tables_[i]; }

private:
  std::vector<Table> tables_;
};

template <class T>
class RangeTraits
  : public RangeTraits<decltype(&std::remove_reference<T>::type::operator())> {
};

template <class C, class R, class... A>
class RangeTraits<R (C::*)(A...) const> {
public:
  typedef R r_type;
};

class ColumnBuilder {
public:
  template <class Callback>
  ColumnBuilder(const std::string &name, Callback &&callback)
    : impl_{std::make_shared<Impl<Callback> >(
        std::move(name), std::forward<Callback>(callback))} {}

  std::unique_ptr<Column> Build(const std::size_t length) const {
    return impl_->Build(length);
  }

private:
  class ImplBase {
  public:
    inline virtual ~ImplBase();
    virtual std::unique_ptr<Column> Build(const std::size_t length) = 0;
  };

  template <class Callable>
  class Impl : public ImplBase {
  public:
    Impl(const std::string &&name, Callable &&callback)
      : name_{std::move(name)}, callback_{std::forward<Callable>(callback)} {}

    std::unique_ptr<Column> Build(const std::size_t length) final {
      auto *column =
        new TypedColumn<RangeTraits<decltype(callback_)>::r_type::value>(name_);
      column->range(length, callback_);
      return std::unique_ptr<Column>(column);
    }

  private:
    const std::string name_;
    Callable          callback_;
  };

  std::shared_ptr<ImplBase> impl_;
};

inline ColumnBuilder::ImplBase::~ImplBase() = default;

class TableBuilder {
public:
  TableBuilder(const std::string &&                 name,
               std::initializer_list<ColumnBuilder> builders)
    : name_{std::move(name)}, builders_{builders} {}

  Table build(const std::size_t length) {  //! \deprecated
    return Build(length);
  }

  Table Build(const std::size_t length) const {
    std::vector<std::shared_ptr<Column> > columns;
    columns.resize(builders_.size());
    std::transform(std::begin(builders_),
                   std::end(builders_),
                   columns.begin(),
                   [length](const ColumnBuilder &builder) {
                     return std::move(builder.Build(length));
                   });
    return Table(name_, std::move(columns));
  }

private:
  const std::string                    name_;
  std::initializer_list<ColumnBuilder> builders_;
};

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
    std::transform(std::begin(builders_),
                   std::end(builders_),
                   tables.begin(),
                   [length](const TableBuilder &builder) -> Table {
                     return builder.Build(length);
                   });
    return TableGroup{tables};
  }

private:
  std::initializer_list<TableBuilder> builders_;
};

using Index = const std::size_t;

template <gdf_dtype U>
std::vector<typename DType<U>::value_type>
HostVectorFrom(gdf_column_cpp &column) {
  std::vector<typename DType<U>::value_type> vector;
  vector.reserve(column.size());
  cudaMemcpy(vector.data(),
             column.data(),
             column.size() * DType<U>::size,
             cudaMemcpyDeviceToHost);
  return vector;
}

}  // namespace utils
}  // namespace test
}  // namespace ral

#endif
