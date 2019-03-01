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
        using TokenValue = typename CommServer::TokenValue;

        using Message = blazingdb::communication::messages::Message;

    public:
        static void start();

        static Server& getInstance();

    private:
        Server();

    public:
        ~Server();

    public:
        void registerContext(const TokenValue& context_token);

    public:
        std::shared_ptr<Message> getMessage(const TokenValue& token_value);

    private:
        Server(Server&&) = delete;

        Server(const Server&) = delete;

        Server& operator=(Server&&) = delete;

        Server& operator=(const Server&) = delete;

    private:
        void setEndPoints();

    private:
        std::thread thread;
        std::shared_ptr<CommServer> comm_server;
    };

} // namespace network
} // namespace communication
} // namespace ral
