#include "communication/network/Client.h"
#include "tests/distribution/mocking/ClientMock.h"

namespace ral {
namespace communication {
namespace network {

std::shared_ptr<Client::Status> Client::send(const Node& node, std::shared_ptr<Message>& message) {
    return ClientMock::getInstance().send(node, message);
}

} // namespace network
} // namespace communication
} // namespace ral
