#pragma once

#include <memory>
#include "blazingdb/communication/network/Status.h"

namespace blazingdb {
namespace communication {
    class Node;
    namespace messages {
        class Message;
    }
}
}

namespace ral {
namespace communication {
namespace network {

    namespace {
        using Node = blazingdb::communication::Node;
        using Message = blazingdb::communication::messages::Message;
    }

    class Client {
    public:
        using Status = blazingdb::communication::network::Status;

    public:
        static std::shared_ptr<Status> send(const Node& node,
                                            std::shared_ptr<Message>& message);

        static std::shared_ptr<Status> sendNodeData(const std::string& orchestratorIp,
                                                    int16_t orchestratorPort,
                                                    std::shared_ptr<Message>& message);
    };

} // namespace network
} // namespace communication
} // namespace ral
