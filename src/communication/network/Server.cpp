#include "communication/network/Server.h"

namespace ral {
namespace communication {
namespace network {

    void Server::start() {
        getInstance();
    }

    Server& Server::getInstance() {
        static Server server;
        return server;
    }

    Server::Server() {
        comm_server = CommServer::Make();
        thread = std::thread([this]() {
            comm_server->Run();
        });
        std::this_thread::yield();
    }

    Server::~Server() {
        comm_server->Close();
        thread.join();
    }

} // namespace network
} // namespace communication
} // namespace ral
