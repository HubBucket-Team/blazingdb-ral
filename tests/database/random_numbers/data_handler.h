#pragma once

#include <vector>
#include <string>
#include <fstream>

namespace bdb {
namespace test {

    struct Data {
        int data{};
        int mask{};
    };

    struct Meta {
        int number_columns{};
        int size_column{};
        int min_value{};
        int max_value{};
    };

    class DataHandler {
    public:
        DataHandler(std::string&& filename);

        DataHandler(const std::string& filename);

        ~DataHandler();

    public:
        const Meta getMeta() const;

    public:
        bool hasNext();

        std::vector<Data> readNext();

    public:
        std::vector<std::vector<Data>> readData();

    private:
        void openfile(const std::string& filename);

    private:
        std::ifstream file;

    private:
        Meta meta;

    private:
        int counter{};
    };

} // namespace bdb
} // namespace test
