#pragma once

#include <gmock/gmock.h>
#include "blazingdb/communication/Node.h"
#include "blazingdb/communication/messages/Message.h"
#include "blazingdb/communication/network/Status.h"
#include "communication/network/Client.h"

namespace {
using blazingdb::communication::Node;
using blazingdb::communication::network::Status;
using blazingdb::communication::messages::Message;
} // namespace

struct ClientMock {
    static ClientMock& getInstance() {
        static ClientMock mock;
        return mock;
    }

    MOCK_METHOD2(send, std::shared_ptr<Status>(const Node&, std::shared_ptr<Message>&));
};