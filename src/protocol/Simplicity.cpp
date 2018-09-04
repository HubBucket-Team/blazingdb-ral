#include <iostream>

#include "Protocol/MessageBuilder.h"

int ads_main(int argc, char *argv[]) {
    const std::string statement = "seelct * from orders";
    const std::string authorization = "TODO"; // TODO percy
    BlazingProtocol::QueryResult queryResult;
    queryResult.sock = 8080;
    queryResult.duration = 123;

    const BlazingProtocol::Payload payload = BlazingProtocol::buildServerMessage(statement, authorization, queryResult);

    return 0;
}
