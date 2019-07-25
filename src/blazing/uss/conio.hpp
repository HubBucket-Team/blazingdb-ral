#ifndef BLAZING_USS_CONIO_HPP_
#define BLAZING_USS_CONIO_HPP_

#include <string>

namespace blazing {
namespace uss {
namespace conio {

// Why this class?
// The point is that we have a logger, custom prints, exceptions texts and
// outputs from tools we use. So, the idea is to have a character stream
// abstracttion in future for all components to print in standard format and
// thus avoid to have a dirty output and make easier reading of logs for QAs,
// Devs, and Uers.
class Console {
public:
    enum Color { kNone, kRed, kGreen };

    Console & Write(const std::string && s);
    Console & SetColor(const Color color);
    Console & EndLine();
};

}  // namespace conio
}  // namespace uss
}  // namespace blazing

#endif
