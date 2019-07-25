#ifndef BLAZING_COMMON_DEFINITION_HPP_
#define BLAZING_COMMON_DEFINITION_HPP_

#define BLAZING_CONST const __attribute__((__const__))
#define BLAZING_DEPRECATED __attribute__((deprecated))
#define BLAZING_INLINE inline __attribute__((always_inline))
#define BLAZING_NOEXPORT __attribute__((visibility("internal")))
#define BLAZING_NORETURN __attribute__((__noreturn__))
#define BLAZING_PURE __attribute__((__pure__))

#define BLAZING_STATIC_LOCAL(Kind, name) static const Kind & name = *new Kind

#define blazing_likely(x) __builtin_expect(x, 1)
#define blazing_unlikely(x) __builtin_expect(x, 0)

#define BLAZING_INTERFACE(Kind)                                                \
public:                                                                        \
    virtual ~Kind() = default;                                                 \
                                                                               \
protected:                                                                     \
    explicit Kind() = default;                                                 \
                                                                               \
private:                                                                       \
    Kind(const Kind &)  = delete;                                              \
    Kind(const Kind &&) = delete;                                              \
    void operator=(const Kind &) = delete;                                     \
    void operator=(const Kind &&) = delete

#define BLAZING_CONCRETE(Kind)                                                 \
private:                                                                       \
    Kind(const Kind &)  = delete;                                              \
    Kind(const Kind &&) = delete;                                              \
    void operator=(const Kind &) = delete;                                     \
    void operator=(const Kind &&) = delete

#define BLAZING_DTO(Kind)                                                      \
    BLAZING_CONCRETE(Kind);                                                    \
                                                                               \
public:                                                                        \
    inline explicit Kind() = default;                                          \
    inline ~Kind()         = default

#define BLAZING_MIXIN(Kind)                                                    \
    BLAZING_CONCRETE(Kind);                                                    \
                                                                               \
protected:                                                                     \
    inline explicit Kind() = default;                                          \
                                                                               \
public:                                                                        \
    inline ~Kind() = default

#endif
