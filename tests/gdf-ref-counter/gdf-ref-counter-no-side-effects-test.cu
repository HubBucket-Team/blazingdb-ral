#include "gdf-ref-counter-test.h"
/*
TEST(GdfRefCounterTest, DISABLE_NoSideEffectsForCounting) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();
  gdf_column  *   column = new gdf_column;

  EXPECT_FALSE(counter->contains_column({column}));
  EXPECT_EQ(0, counter->get_map_size());
  EXPECT_NO_THROW({ counter->decrement(column); });

  EXPECT_FALSE(counter->contains_column({column}));
  EXPECT_EQ(0, counter->get_map_size());
  EXPECT_NO_THROW({ counter->increment(column); });

  EXPECT_FALSE(counter->contains_column({column}));
  EXPECT_EQ(0, counter->get_map_size());
}

TEST(GdfRefCounterTest, DISABLE_NoSideEffectsForRegisteringNull) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();

  gdf_column * ptrs{nullptr};

  counter->register_column(nullptr);
  EXPECT_FALSE(counter->contains_column(ptrs));
  EXPECT_EQ(0, counter->get_map_size());
}

TEST(GdfRefCounterTest, DISABLE_NoSideEffectsForDeregisteringNull) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();

  EXPECT_EQ(0, counter->get_map_size());
  counter->deregister_column(nullptr);
  EXPECT_EQ(0, counter->get_map_size());
}

TEST(GdfRefCounterTest, DISABLE_NoSideEffectsForDeregistering) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();

  gdf_column column;

  counter->deregister_column(column);
  EXPECT_FALSE(counter->contains_column({column}));
  EXPECT_EQ(0, counter->get_map_size());

  delete column;
}
*/
