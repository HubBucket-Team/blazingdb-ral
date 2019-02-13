#include <fstream>
#include "database/random_numbers/data_handler.h"

namespace bdb {
namespace test {

    DataHandler::DataHandler(std::string&& filename)
    {
        openfile(filename);
    }

    DataHandler::DataHandler(const std::string& filename)
    {
        openfile(filename);
    }

    DataHandler::~DataHandler() {
        file.close();
    }

    const Meta DataHandler::getMeta() const {
        return meta;
    }

    bool DataHandler::hasNext() {
        return (counter < meta.number_columns);
    }

    std::vector<Data> DataHandler::readNext() {
        counter++;
        char delimiter{};
        std::vector<Data> data(meta.size_column);
        for (std::size_t x = 0; x < meta.size_column; ++x) {
            file >> data[x].data;
            if (x < (meta.size_column - 1)) {
                file >> delimiter;
            }
        }
        for (std::size_t x = 0; x < meta.size_column; ++x) {
            file >> data[x].mask;
            if (x < (meta.size_column - 1)) {
                file >> delimiter;
            }
        }
        return data;
    }

    std::vector<std::vector<Data>> DataHandler::readData() {
        std::vector<std::vector<Data>> data;
        data.reserve(meta.number_columns);
        while (hasNext()) {
            data.emplace_back(readNext());
        }
        return data;
    }

    void DataHandler::openfile(const std::string& filename) {
        file.open(filename, std::ios::in);

        file >> meta.number_columns;
        file >> meta.size_column;
        file >> meta.min_value;
        file >> meta.max_value;
    }

} // namespace bdb
} // namespace test
