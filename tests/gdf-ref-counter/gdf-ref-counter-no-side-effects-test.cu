#include "gdf-ref-counter-test.h"

TEST(GdfRefCounterTest, NoSideEffectsForCounting) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();
  gdf_column     column;

  EXPECT_FALSE(counter->contains_column({column.data, column.valid}));
  EXPECT_EQ(0, counter->get_map_size());
  EXPECT_NO_THROW({ counter->decrement(&column); });

  EXPECT_FALSE(counter->contains_column({column.data, column.valid}));
  EXPECT_EQ(0, counter->get_map_size());
  EXPECT_NO_THROW({ counter->increment(&column); });

  EXPECT_FALSE(counter->contains_column({column.data, column.valid}));
  EXPECT_EQ(0, counter->get_map_size());
}

TEST(GdfRefCounterTest, NoSideEffectsForRegisteringNull) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();

  rc_key_t ptrs{nullptr, nullptr};

  counter->register_column(nullptr);
  EXPECT_FALSE(counter->contains_column(ptrs));
  EXPECT_EQ(0, counter->get_map_size());
}

TEST(GdfRefCounterTest, NoSideEffectsForDeregisteringNull) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();

  EXPECT_EQ(0, counter->get_map_size());
  counter->deregister_column(nullptr);
  EXPECT_EQ(0, counter->get_map_size());
}

TEST(GdfRefCounterTest, NoSideEffectsForDeregistering) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();

  gdf_column column;

  counter->deregister_column(&column);
  EXPECT_FALSE(counter->contains_column({column.data, column.valid}));
  EXPECT_EQ(0, counter->get_map_size());
}
