#include "gdf-ref-counter-test.h"
/*
TEST(GdfRefCounterTest, CountingRefs) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();
  gdf_column *    column = new gdf_column;

  counter->register_column(column);
  EXPECT_TRUE(counter->contains_column({column}));
  EXPECT_EQ(1, counter->get_map_size());

  counter->increment(column);
  EXPECT_TRUE(counter->contains_column({column}));
  EXPECT_EQ(1, counter->get_map_size());

  counter->decrement(column);
  EXPECT_TRUE(counter->contains_column({column}));
  EXPECT_EQ(1, counter->get_map_size());

  counter->decrement(column);
  EXPECT_FALSE(counter->contains_column({column}));
  EXPECT_EQ(0, counter->get_map_size());

  delete column;
}
*/
