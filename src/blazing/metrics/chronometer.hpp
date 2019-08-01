#ifndef BLAZING_METRICS_CHRONOMETER_HPP_
#define BLAZING_METRICS_CHRONOMETER_HPP_

#include <memory>

#include <blazing/common/definition.hpp>

namespace blazing {
namespace metrics {

class IllegalStateException : public std::exception {
public:
    virtual ~IllegalStateException() noexcept = default;

    virtual const char * what() const noexcept {
        static const char what[] = "Illegal State Exception";
        return what;
    };
};

class Check {
public:
    static void State(bool expressionValue);
};

class TimeUnit {
public:
    enum type { kMicroSeconds, kNanoSeconds };
};

class Watch {
    BLAZING_INTERFACE(Watch);

public:
    virtual std::uintmax_t Read() const noexcept = 0;

    static std::unique_ptr<Watch> InternalWatch() noexcept;
};

// Like chronometer in blazing-calcite
class Chronometer {
    BLAZING_INTERFACE(Chronometer);

public:
    virtual bool IsRunning() const noexcept = 0;

    virtual Chronometer & Start() = 0;
    virtual Chronometer & Stop()  = 0;

    virtual std::uintmax_t Elapsed() const noexcept                     = 0;
    virtual std::uintmax_t Elapsed(const TimeUnit::type) const noexcept = 0;

    // TODO: for compile time call
    // template <class TimeUnit> std::uintmax_t Elapsed();

    virtual Chronometer & Reset() noexcept = 0;

    static std::unique_ptr<Chronometer> MakeUnstarted();
    static std::unique_ptr<Chronometer> MakeStarted();
    static std::unique_ptr<Chronometer> Content(const Watch & watch);
};

}  // namespace metrics
}  // namespace blazing

#endif
