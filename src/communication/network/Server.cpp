#include "communication/network/Server.h"
#include "communication/messages/ComponentMessages.h"

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

        setEndPoints();

        thread = std::thread([this]() {
            comm_server->Run();
        });
        std::this_thread::yield();
    }

    Server::~Server() {
        comm_server->Close();
        thread.join();
    }

    void Server::registerContext(const Server::TokenValue& context_token) {
        comm_server->registerContext(context_token);
    }

    std::shared_ptr<Server::Message> Server::getMessage(const TokenValue& token_value) {
        return comm_server->getMessage(token_value);
    }

    void Server::setEndPoints() {
        namespace messages = ral::communication::messages;

        // message SampleToNodeMasterMessage
        {
            const std::string endpoint = messages::SampleToNodeMasterMessage::getMessageID();
            comm_server->registerEndPoint(endpoint, CommServer::Methods::Post);
            comm_server->registerDeserializer(endpoint, ral::communication::messages::SampleToNodeMasterMessage::Make);
        }

        // message PartitionPivotsMessage
        {
            const std::string endpoint = messages::PartitionPivotsMessage::getMessageID();
            comm_server->registerEndPoint(endpoint, CommServer::Methods::Post);
            comm_server->registerDeserializer(endpoint, messages::PartitionPivotsMessage::Make);
        }

        // message PartitionPivotsMessage
        {
            const std::string endpoint = messages::DataScatterMessage::getMessageID();
            comm_server->registerEndPoint(endpoint, CommServer::Methods::Post);
            comm_server->registerDeserializer(endpoint, messages::DataScatterMessage::Make);
        }
    }

} // namespace network
} // namespace communication
} // namespace ral
