#include <blazing/metrics/chronometer.hpp>

#include <atomic>

namespace blazing {
namespace metrics {
namespace testing {

class StubWatch : public Watch {
    BLAZING_CONCRETE(StubWatch);

public:
    explicit StubWatch() = default;

    std::uintmax_t Read() const noexcept final { return time_; }

    StubWatch & Add(const std::uintmax_t increment /*, timeUnit*/) {
        time_.fetch_add(increment);
        return *this;
    }

    StubWatch & SetStep(const std::uintmax_t value /*, timeUnit*/) {
        time_.store(value);
        return *this;
    }

    static std::unique_ptr<StubWatch> Make() {
        return std::make_unique<StubWatch>();
    }

private:
    std::atomic_uintmax_t   time_;
    volatile std::uintmax_t step_;
};

}  // namespace testing
}  // namespace metrics
}  // namespace blazing
