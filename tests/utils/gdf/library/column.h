#pragma once

#include <cassert>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include <GDFColumn.cuh>
#include <gdf/gdf.h>

#include "any.h"
#include "definitions.h"
#include "types.h"
#include "vector.h"

namespace gdf {
namespace library {

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
  Column(const std::string &name) : name_{name} {}

  virtual ~Column();

  virtual gdf_column_cpp ToGdfColumnCpp() const = 0;

  virtual const void *get(const std::size_t i) const = 0;

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

  Wrapper operator[](const std::size_t i) const { return Wrapper{i, this}; }

  virtual size_t      size() const                      = 0;
  virtual size_t      print(std::ostream &stream) const = 0;
  virtual std::string get_as_str(int index) const       = 0;

  const std::string &name() const { return name_; }

protected:
  static gdf_column_cpp Create(const gdf_dtype   dtype,
                               const std::size_t length,
                               const void *      data,
                               const std::size_t size);

protected:
  const std::string name_;
};

Column::~Column() {}

gdf_column_cpp Column::Create(const gdf_dtype   dtype,
                              const std::size_t length,
                              const void *      data,
                              const std::size_t size) {
  gdf_column_cpp column_cpp;
  column_cpp.create_gdf_column(dtype, length, const_cast<void *>(data), size);
  return column_cpp;
}

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

template <gdf_dtype DType>
class TypedColumn : public Column {
public:
  using value_type = typename DTypeTraits<DType>::value_type;
  using Callback   = std::function<value_type(const std::size_t)>;

  TypedColumn(const std::string &name) : Column(name) {}

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
    return Create(DType, this->size(), values_.data(), sizeof(value_type));
  }

  value_type operator[](const std::size_t i) const { return values_[i]; }

  const void *get(const std::size_t i) const final { return &values_[i]; }

  void FillData(std::vector<value_type> values) {
    for (std::size_t i = 0; i < values.size(); i++) {
      values_.push_back(values.at(i));
    }
  }

  size_t size() const final { return values_.size(); }

  size_t print(std::ostream &stream) const final {
    for (std::size_t i = 0; i < values_.size(); i++) {
      stream << values_.at(i) << " | ";
    }
  }

  std::string get_as_str(int index) const final {
    std::ostringstream out;
    if (std::is_floating_point<value_type>::value) { out.precision(1); }
    out << std::fixed << values_.at(index);
    return out.str();
  }

private:
  std::basic_string<value_type> values_;
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

  ColumnBuilder() {}
  ColumnBuilder &operator=(const ColumnBuilder &other) {
    impl_ = other.impl_;
    return *this;
  }

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

template <gdf_dtype id>
class Literals {
public:
  using value_type       = typename DType<id>::value_type;
  using initializer_list = std::initializer_list<value_type>;

  Literals(initializer_list values) : values_{values} {}

  initializer_list values() const { return values_; }

  std::size_t size() const { return values_.size(); }

private:
  std::initializer_list<typename DType<id>::value_type> values_;
};

class LiteralColumnBuilder {
public:
  template <gdf_dtype id>
  LiteralColumnBuilder(const std::string &name, Literals<id> values)
    : impl_{std::make_shared<Impl<id> >(name, values)} {}

  std::unique_ptr<Column> Build() const { return impl_->Build(); }

  operator ColumnBuilder() const { return *impl_; }

  std::size_t length() const { return impl_->length(); }

private:
  class ImplBase {
  public:
    inline virtual ~ImplBase();
    virtual std::unique_ptr<Column> Build()                        = 0;
    virtual                         operator ColumnBuilder() const = 0;
    virtual std::size_t             length() const                 = 0;
  };

  template <gdf_dtype id>
  class Impl : public ImplBase {
  public:
    Impl(const std::string &name, Literals<id> literals)
      : name_{name}, literals_{literals},
        builder_{ColumnBuilder(name_, [this](Index i) -> DType<id> {
          return *(literals_.values().begin() + static_cast<std::ptrdiff_t>(i));
        })} {}

    std::unique_ptr<Column> Build() { return builder_.Build(literals_.size()); }

    operator ColumnBuilder() const { return builder_; }

    virtual std::size_t length() const { return literals_.size(); }

  private:
    const std::string name_;
    Literals<id>      literals_;
    ColumnBuilder     builder_;
  };

  std::shared_ptr<ImplBase> impl_;
};

inline LiteralColumnBuilder::ImplBase::~ImplBase() = default;

class ColumnFiller {
public:
  template <class Type>
  ColumnFiller(const std::string &name, const std::vector<Type> &values) {
    auto *pointer = new TypedColumn<GdfDataType<Type>::Value>(name);
    pointer->FillData(values);
    column_ = std::shared_ptr<Column>(pointer);
  }
  ColumnFiller() = default;
  ColumnFiller(const ColumnFiller &other) = default;
  ColumnFiller &operator=(const ColumnFiller &other) = default;

  std::shared_ptr<Column> Build() const { return column_; }

private:
  std::shared_ptr<Column> column_;
};

}  // namespace library
}  // namespace gdf
