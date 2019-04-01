#include "communication/network/Client.h"
#include "blazingdb/communication/network/Client.h"

namespace ral {
namespace communication {
namespace network {

    std::shared_ptr<Client::Status> Client::send(const Node& node,
                                                 std::shared_ptr<Message>& message) {
        auto client = blazingdb::communication::network::Client::Make();
        return client->send(node, message);
    }

    std::shared_ptr<Client::Status> Client::sendNodeData(const std::string& orchestratorIp,
                                                        int16_t orchestratorPort,
                                                        std::shared_ptr<Message>& message) {
        auto client = blazingdb::communication::network::Client::Make();
        return client->SendNodeData(orchestratorIp, orchestratorPort, *message);
    }

} // namespace network
} // namespace communication
} // namespace ral
