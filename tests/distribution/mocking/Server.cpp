#include "communication/network/Server.h"
#include "tests/distribution/mocking/ServerMock.h"

namespace ral {
namespace communication {
namespace network {

Server& Server::getInstance() {
    static Server server;
    return server;
}

Server::Server() {
}

Server::~Server() {
}

std::shared_ptr<Server::Message> Server::getMessage(const ContextToken& context_token) {
    return ServerMock::getInstance().getMessage(context_token);
}

} // namespace network
} // namespace communication
} // namespace ral
