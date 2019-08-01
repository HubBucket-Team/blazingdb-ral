#include "conio.hpp"

#include <iostream>
#include <unordered_map>

#include <blazing/common/definition.hpp>

namespace blazing {
namespace uss {
namespace conio {

Console & Console::Write(const std::string && s) {
    std::cout << s;
    return *this;
}

namespace {
class BLAZING_NOEXPORT ColorHash {
public:
    template <class T>
    std::size_t operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};
}  // namespace

Console & Console::SetColor(const Color color) {
    static std::unordered_map<Color, std::string, ColorHash> codeOf{
        {kNone, "0"}, {kRed, "31"}, {kGreen, "32"}};
    std::cout << "\033[" << codeOf[color] << "m";
    return *this;
}

Console & Console::EndLine() {
    std::cout << std::endl;
    return *this;
}

}  // namespace conio
}  // namespace uss
}  // namespace blazing
