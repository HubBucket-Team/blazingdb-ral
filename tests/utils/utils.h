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
  ~Column();

  virtual gdf_column_cpp ToGdfColumnCpp() const = 0;
  virtual const void *   get_values() const     = 0;

  template <gdf_dtype DType>
  typename DTypeTraits<DType>::value_type get(const std::size_t i) const {
    return (*reinterpret_cast<
            const std::basic_string<typename DTypeTraits<DType>::value_type> *>(
      get_values()))[i];
  }

protected:
  static gdf_column_cpp Create(const gdf_dtype   dtype,
                               const std::size_t length,
                               const void *      data,
                               const std::size_t size);
};

template <gdf_dtype DType>
class TypedColumn : public Column {
public:
  using value_type = typename DTypeTraits<DType>::value_type;
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
    return Create(DType, length_, values_.data(), sizeof(value_type));
  }

  const void *get_values() const final { return &values_; }

private:
  const std::string name_;

  std::size_t                   length_;
  std::basic_string<value_type> values_;
};

class Table {
public:
  Table(const std::string &name) : name_{name} {}

  Table(const std::string &                     name,
        std::vector<std::shared_ptr<Column> > &&columns)
    : name_{name}, columns_{std::move(columns)} {}

  const Column &operator[](const std::size_t i) const { return *columns_[i]; }

private:
  const std::string name_;

  std::vector<std::shared_ptr<Column> > columns_;
};

using BlazingFrame = std::vector<std::vector<gdf_column> >;

class TableGroup {
public:
  TableGroup(std::initializer_list<Table> tables) {
    for (Table table : tables) { tables_.push_back(table); }
  }

  TableGroup(std::initializer_list<std::initializer_list<gdf_dtype> > dtypess) {
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

template <gdf_dtype DType>
class Ret {
public:
  static constexpr gdf_dtype dtype = DType;

  using value_type = typename DTypeTraits<DType>::value_type;

  template <class T>
  Ret(const T value) : value_{value} {}

  operator value_type() const { return value_; }

private:
  const value_type value_;
};

class ColumnBuilder {
public:
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
        new TypedColumn<RangeTraits<decltype(callback_)>::r_type::dtype>(name_);
      column->range(length, callback_);
      return std::unique_ptr<Column>(column);
    }

  private:
    const std::string name_;
    Callable          callback_;
  };

  template <class Callback>
  ColumnBuilder(const std::string &name, Callback &&callback)
    : impl_{std::make_shared<Impl<Callback> >(
        std::move(name), std::forward<Callback>(callback))} {}

  std::unique_ptr<Column> Build(const std::size_t length) {
    return impl_->Build(length);
  }

private:
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

  Table Build(const std::size_t length) {
    std::vector<std::shared_ptr<Column> > columns;
    columns.reserve(builders_.size());
    std::transform(builders_.begin(),
                   builders_.end(),
                   columns.begin(),
                   [length](ColumnBuilder &builder) {
                     return std::move(builder.Build(length));
                   });
    return Table(name_, std::move(columns));
  }

private:
  const std::string          name_;
  std::vector<ColumnBuilder> builders_;
};

}  // namespace utils
}  // namespace test
}  // namespace ral

#endif
