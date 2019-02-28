#pragma once

#include <thread>
#include "blazingdb/communication/network/Server.h"

namespace ral {
namespace communication {
namespace network {

    namespace {
        using CommServer = blazingdb::communication::network::Server;
    }

    class Server {
    public:
        static void start();

        static Server& getInstance();

    private:
        Server();

    public:
        ~Server();

    public:
        template <typename Message>
        std::shared_ptr<Message> getMessage() {
            return comm_server->GetMessage<Message>();
        }

    private:
        Server(Server&&) = delete;

        Server(const Server&) = delete;

        Server& operator=(Server&&) = delete;

        Server& operator=(const Server&) = delete;

    private:
        std::thread thread;
        std::shared_ptr<CommServer> comm_server;
    };

} // namespace network
} // namespace communication
} // namespace ral
