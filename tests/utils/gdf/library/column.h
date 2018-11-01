#pragma once

#include <cassert>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "any.h"
#include "definitions.h"
#include "hd.h"
#include "types.h"
#include "vector.h"

namespace gdf {
namespace library {

class Column {
public:
  Column(const std::string &name) : name_{name} {}

  virtual ~Column();

  virtual gdf_column_cpp ToGdfColumnCpp() const = 0;

  virtual const void *get(const std::size_t i) const = 0;

  bool operator==(const Column &other) const {
    for (std::size_t i = 0; i < size(); i++) {
      if ((*static_cast<const std::uint64_t *>(get(i)))
          != (*static_cast<const std::uint64_t *>(other.get(i)))) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const Column &other) const { return !(*this == other); }

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

  virtual size_t size() const = 0;

  virtual std::string to_string() const = 0;

  virtual std::string get_as_str(int index) const = 0;

  std::string name() const { return name_; }

protected:
  static gdf_column_cpp Create(const std::string &name,
                               const gdf_dtype    dtype,
                               const std::size_t  length,
                               const void *       data,
                               const std::size_t  size);

protected:
  const std::string name_;
};

Column::~Column() {}

gdf_column_cpp Column::Create(const std::string &name,
                              const gdf_dtype    dtype,
                              const std::size_t  length,
                              const void *       data,
                              const std::size_t  size) {
  gdf_column_cpp column_cpp;
  column_cpp.create_gdf_column(dtype, length, const_cast<void *>(data), size);
  column_cpp.column_name = name;
  return column_cpp;
}

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
    // assert(end > begin);  // TODO(gcca): bug with gdf_column_cpps
    values_.reserve(end - begin);
    for (std::size_t i = begin; i < end; i++) {
      values_.push_back(callback(i));
    }
  }

  void range(const std::size_t end, Callback callback) {
    range(0, end, callback);
  }

  gdf_column_cpp ToGdfColumnCpp() const final {
    return Create(
      name(), DType, this->size(), values_.data(), sizeof(value_type));
  }

  value_type operator[](const std::size_t i) const { return values_[i]; }

  const void *get(const std::size_t i) const final { return &values_[i]; }

  void FillData(std::vector<value_type> values) {
    for (std::size_t i = 0; i < values.size(); i++) {
      values_.push_back(values.at(i));
    }
  }

  size_t size() const final { return values_.size(); }

  std::string to_string() const final {
    std::ostringstream stream;

    for (std::size_t i = 0; i < values_.size(); i++) {
      stream << values_.at(i) << ",";
    }
    return std::string{stream.str()};
  }

  std::string get_as_str(int index) const final {
    std::ostringstream out;
    if (std::is_floating_point<value_type>::value) { out.precision(1); }
    if (sizeof(value_type) == 1) {
      out << std::fixed << (int) values_.at(index);
    } else {
      out << std::fixed << values_.at(index);
    }
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
  using vector           = std::vector<value_type>;

  Literals(const vector &&values) : values_{std::move(values)} {}
  Literals(const initializer_list &values) : values_{values} {}

  const vector &values() const { return values_; }

  std::size_t size() const { return values_.size(); }

  value_type operator[](const std::size_t i) const { return values_[i]; }

private:
  vector values_;
};

class LiteralColumnBuilder {
public:
  template <gdf_dtype id>
  LiteralColumnBuilder(const std::string &name, Literals<id> values)
    : impl_{std::make_shared<Impl<id> >(name, values)} {}

  LiteralColumnBuilder() {}
  LiteralColumnBuilder &operator=(const LiteralColumnBuilder &other) {
    impl_ = other.impl_;
    return *this;
  }

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

class GdfColumnCppColumnBuilder {
public:
  GdfColumnCppColumnBuilder(const std::string &   name,
                            const gdf_column_cpp &column_cpp)
    : name_{name}, column_cpp_{column_cpp} {}

  GdfColumnCppColumnBuilder()        = default;
  GdfColumnCppColumnBuilder &operator=(const GdfColumnCppColumnBuilder &other) {
    name_       = other.name_;
    column_cpp_ = other.column_cpp_;
    return *this;
  }

  operator LiteralColumnBuilder() const {
    auto column_cpp = const_cast<gdf_column_cpp &>(column_cpp_);
    switch (column_cpp.dtype()) {
#define CASE(D)                                                                \
  case GDF_##D:                                                                \
    return LiteralColumnBuilder {                                              \
      name_, Literals<GDF_##D> {                                               \
        std::move(HostVectorFrom<GDF_##D>(column_cpp))                         \
      }                                                                        \
    }
      CASE(INT8);
      CASE(INT16);
      CASE(INT32);
      CASE(INT64);
      CASE(UINT8);
      CASE(UINT16);
      CASE(UINT32);
      CASE(UINT64);
      CASE(FLOAT32);
      CASE(FLOAT64);
      CASE(DATE32);
      CASE(DATE64);
      CASE(TIMESTAMP);
#undef CASE
    default: throw std::runtime_error("Bad DType");
    }
  }

  std::unique_ptr<Column> Build() const {
    return static_cast<LiteralColumnBuilder>(*this).Build();
  }

private:
  std::string    name_;
  gdf_column_cpp column_cpp_;
};

class ColumnFiller {
public:
  template <class Type>
  ColumnFiller(const std::string &name, const std::vector<Type> &values) {
    auto *pointer = new TypedColumn<GdfDataType<Type>::Value>(name);
    pointer->FillData(values);
    column_ = std::shared_ptr<Column>(pointer);
  }
  ColumnFiller()                          = default;
  ColumnFiller(const ColumnFiller &other) = default;
  ColumnFiller &operator=(const ColumnFiller &other) = default;

  std::shared_ptr<Column> Build() const { return column_; }

private:
  std::shared_ptr<Column> column_;
};

}  // namespace library
}  // namespace gdf
