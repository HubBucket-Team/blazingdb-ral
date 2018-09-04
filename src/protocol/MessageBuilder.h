/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2017 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
 */

#ifndef MESSAGEBUILDER_H_
#define MESSAGEBUILDER_H_

#include <cstdint> // Integer types: uint8_t and uint32_t
#include <string>
#include <vector>

namespace BlazingProtocol {

    // QueryResult holds the query output of QueryPartitionUtil::writeQueryOutputToSocket.
    // Also QueryResult is the input to create the ResultSet (part of the ServerMessage)
    struct QueryResult {
        std::vector<long long> limits;
        int sock;
        double duration;
    };

    // Payload should not manage the destruction of data, is just a data info holder
    class Payload {
        public:
            Payload(uint8_t* data = nullptr, uint32_t size = 0);

            const uint8_t* data() const {
                return m_data;
            }

            const uint32_t size() const {
                return m_size;
            }

        private:
            uint8_t* m_data;
            uint32_t m_size;
    };

    const Payload buildServerMessage(const std::string &statement, const std::string &authorization, const QueryResult &queryResult);


    //const Payload buildServerMessage(gdf_columns[][]);

    int ads_main();
}

#endif /* MESSAGEBUILDER_H_ */
