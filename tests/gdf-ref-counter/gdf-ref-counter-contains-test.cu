#include "gdf-ref-counter-test.h"

TEST(GdfRefCounterTest, ContainsInEmpty) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();
  rc_key_t ptrs{nullptr, nullptr};

  EXPECT_FALSE(counter->contains_column(ptrs));
  EXPECT_EQ(0, counter->get_map_size());
}
