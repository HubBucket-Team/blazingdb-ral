#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <mutex>
#include <deque>
#include "blazingdb/communication/Context.h"
#include "blazingdb/communication/network/Status.h"
#include "distribution/Exception.h"
#include "distribution/NodeColumns.h"
#include "communication/CommunicationData.h"
#include "communication/factory/MessageFactory.h"
#include "communication/network/Client.h"
#include "communication/network/Server.h"
#include "operators/JoinOperator.h"
#include "Traits/RuntimeTraits.h"
#include "utilities/RalColumn.h"
#include "tests/utilities/column/column_utilities.h"
#include "tests/utilities/gdf_column_cpp_utilities.h"

// gtest
using ::testing::_;
using ::testing::InvokeWithoutArgs;

// blazingdb communication library
using blazingdb::communication::Node;
using blazingdb::communication::Context;
using blazingdb::communication::ContextToken;
using blazingdb::communication::network::Status;
using blazingdb::communication::messages::Message;


struct DistributedJoinTest : public ::testing::Test {
    DistributedJoinTest() {
        initSelfNode();
        createContext();
        initializeCommunicationData();
    }

    ~DistributedJoinTest() {
    }

    void SetUp() override {
    }

    void TearDown() override {
    }

    void initSelfNode() {
        self_node_ = Node::makeShared(0, self_ip_, self_port_);
    }

    void createContext() {
        // Create master node
        master_node_ = Node::makeShared(12, "192.168.0.2", 1012);

        // Create worker nodes
        std::vector<std::shared_ptr<Node>> nodes;
        nodes.push_back(master_node_);
        nodes.push_back(self_node_);

        // Create logical plan
        std::string logical_plan {"some logical plan"};

        // Create context
        context_ = std::make_shared<Context>(nodes, master_node_, logical_plan);
    }

    void initializeCommunicationData() {
        using ral::communication::CommunicationData;
        auto& communication_data = CommunicationData::getInstance();
        communication_data.initialize(0, "192.168.0.100", 1000, self_ip_, self_port_);
    }

    std::shared_ptr<Context> context_;
    std::shared_ptr<Node> master_node_;

    const std::string self_ip_{"192.168.0.10"};
    const unsigned short self_port_{7897};
    std::shared_ptr<Node> self_node_;
};


struct ClientMock {
    static ClientMock& getInstance() {
        static ClientMock mock;
        return mock;
    }

    MOCK_METHOD2(send, std::shared_ptr<Status>(const Node&, std::shared_ptr<Message>&));
};


struct ServerFake {
    static ::testing::NiceMock<ServerFake>& getInstance() {
        static ::testing::NiceMock<ServerFake> fake;
        return fake;
    }

    MOCK_METHOD1(getMessage, std::shared_ptr<Message>(const ContextToken&));

    std::shared_ptr<Message> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        auto message = queue_.front();
        queue_.pop_front();
        return message;
    }

    void push(std::shared_ptr<Message>&& table) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push_back(table);
    }

    void delegate_on_call() {
        ON_CALL(*this, getMessage(_))
               .WillByDefault(InvokeWithoutArgs(&ServerFake::getInstance(), &ServerFake::pop));
    }

    std::mutex mutex_;
    std::deque<std::shared_ptr<Message>> queue_;
};


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
    return ServerFake::getInstance().getMessage(context_token);
}

std::shared_ptr<Client::Status> Client::send(const Node& node, std::shared_ptr<Message>& message) {
    return ClientMock::getInstance().send(node, message);
}

} // namespace network
} // namespace communication
} // namespace ral


TEST_F(DistributedJoinTest, SimpleDistributedJoinWithoutBitmask) {
    // Query string
    std::string query { "LogicalJoin(condition=[AND(=($0, $3), =($1, $4))], joinType=[inner])" };

    // Get data for validation
    const auto& context_token = context_->getContextToken();

    // Generate data
    constexpr auto DType = GDF_INT32;
    using Type = ral::traits::type<DType>;

    std::vector<std::vector<Type>> local_lhs = {
        { 1,  2,  3,  4,  5,  6,  7,  8},
        {21, 22, 23, 24, 25, 26, 27, 28},
        {31, 32, 33, 34, 35, 36, 37, 38}
    };

    std::vector<std::vector<Type>> local_rhs = {
        { 1, 10,  3,  4, 12, 13,  7, 15},
        {21, 22, 11, 24, 25, 14, 27, 16},
        {41, 42, 43, 44, 45, 46, 47, 48},
        {51, 52, 53, 54, 55, 56, 57, 58}
    };

    std::vector<std::vector<Type>> remote_lhs = {
        {14, 16, 17, 18},
        {15, 26, 27, 28},
        {65, 66, 67, 68}
    };

    std::vector<std::vector<Type>> remote_rhs = {
        {3,  2,  1},
        {23, 22, 21},
        {73, 72, 71},
        {83, 82, 81},
    };

    blazing_frame frame;
    frame.add_table(ral::test::create_table<DType>(local_lhs));
    frame.add_table(ral::test::create_table<DType>(local_rhs));

    std::vector<gdf_column_cpp> table_remote_lhs = ral::test::create_table<DType>(remote_lhs);
    std::vector<gdf_column_cpp> table_remote_rhs = ral::test::create_table<DType>(remote_rhs);

    // Generate remote messages
    using MessageFactory = ral::communication::messages::Factory;
    auto message_remote_lhs = MessageFactory::createColumnDataMessage(context_token, *self_node_, table_remote_lhs);
    auto message_remote_rhs = MessageFactory::createColumnDataMessage(context_token, *self_node_, table_remote_rhs);

    // Configure fake
    auto& server_fake = ServerFake::getInstance();
    server_fake.push(std::move(message_remote_lhs));
    server_fake.push(std::move(message_remote_rhs));
    server_fake.delegate_on_call();

    // Expect call to Client::send
    EXPECT_CALL(ClientMock::getInstance(), send(_, _))
               .Times(2);

    // Execute test
    blazing_frame output_frame = ral::operators::process_join(context_.get(), frame, query);

    // Generate expected output
    std::vector<std::vector<Type>> join_output = {
        { 1,  2},
        {21, 22},
        {31, 32},
        { 1,  2},
        {21, 22},
        {71, 72},
        {81, 82}
    };

    // Verify
    auto& output_table = output_frame.get_columns()[0];
    for (std::size_t k = 0; k < output_table.size(); ++k) {
        auto output_data = ral::test::get_column_data<Type>(output_table[k]);

        for (std::size_t i = 0; i < output_data.size(); ++i) {
            ASSERT_EQ(output_data[i], join_output[k][i]);
        }
    }
}