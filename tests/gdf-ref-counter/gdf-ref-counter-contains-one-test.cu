#include "gdf-ref-counter-test.h"
/*
TEST(GdfRefCounterTest, ContainsOneElement) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();
  gdf_column  * column = new gdf_column;

  EXPECT_FALSE(counter->contains_column({column}));
  EXPECT_EQ(0, counter->get_map_size());

  counter->register_column(column);

  EXPECT_TRUE(counter->contains_column({column}));
  EXPECT_EQ(1, counter->get_map_size());

  delete column;
}
*/
