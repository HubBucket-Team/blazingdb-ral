#pragma once

#include <thread>
#include "blazingdb/communication/ContextToken.h"
#include "blazingdb/communication/network/Server.h"

namespace ral {
namespace communication {
namespace network {

    namespace {
        using CommServer = blazingdb::communication::network::Server;
        using ContextToken = blazingdb::communication::ContextToken;
    }

    class Server {
    public:
        using Message = blazingdb::communication::messages::Message;

    public:
        static void start(unsigned short port = 8000);

        static Server& getInstance();

    private:
        Server();

    public:
        ~Server();

    public:
        void registerContext(const ContextToken& context_token);

    public:
        std::shared_ptr<Message> getMessage(const ContextToken& token_value);

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

    private:
        static unsigned short port_;
    };

} // namespace network
} // namespace communication
} // namespace ral
