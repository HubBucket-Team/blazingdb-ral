
#pragma once

#include <string>
#include <vector>

struct Item {
  std::string query;
  std::string logicalPlan;
  std::vector<std::string> dataTypes;
  std::vector<std::string> resultTypes;
  std::vector<std::vector<std::string> > data;
  std::vector<std::vector<std::string> > result;
};

std::vector<Item> inputSet{
Item{"select (manaLevel / 2) from heros", "Plan(Fake)\n  Scan(heroes)\n    Projection(1)\n", {"gdf_INT32","gdf_FLOAT"}, {"gdf_DOUBLE"}, {{"1","2"},{"0.3","0.2"}}, {{"0.15"},{"0.1"}}},
Item{"select (heroNumber * power) from heros", "Plan(Fake)\n  Scan(heroes)\n    Projection(1)\n", {"gdf_DOUBLE","gdf_INT64"}, {"gdf_DOUBLE"}, {{"0.111","0.222"},{"2","3"}}, {{"0.111"},{"0.444"}}},
};

