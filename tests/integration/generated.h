#pragma once

#include <string>
#include <vector>

struct Item {
<<<<<<< HEAD
    std::string query;
    std::string logicalPlan;
    std::vector<std::string> dataTypes;
    std::vector<std::string> resultTypes;
    std::vector<std::vector<std::string> > data;
    std::vector<std::vector<std::string> > result;
};

std::vector<Item> inputSet = {
    Item {"select * from main.emps", "Plan(Fake)\n  Scan(heroes)\n    Projection(1)\n", {"GDF_INT8","GDF_INT8"}, {"GDF_INT8","GDF_INT8"}, {{"1","2","3","4","5","6","7","8","9","10"},{"10","20","30","40","50","60","70","80","90","100"}}, {{"1","2","3","4","5","6","7","8","9","10"},{"10","20","30","40","50","60","70","80","90","100"}}},
    Item {"select id > 3 from main.emps", "Plan(Fake)\n  Scan(heroes)\n    Projection(1)\n", {"GDF_INT8","GDF_INT8"}, {"GDF_INT8"}, {{"1","2","3","4","5","6","7","8","9","10"},{"10","20","30","40","50","60","70","80","90","100"}}, {{"4","5","6","7","8","9","10"}}}
    // ... más items
};
