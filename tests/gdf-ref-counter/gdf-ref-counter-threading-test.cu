#include <thread>
#include <vector>

#include "gdf-ref-counter-test.h"

TEST(GdfRefCounterTest, Threading) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();
  gdf_column     column;

  const std::size_t length = 100;

  std::vector<std::shared_ptr<std::thread> > threads;

  for (std::size_t i = 0; i < length; i++) {
    auto thread = std::make_shared<std::thread>(
      [&counter, &column]() { counter->register_column(&column); });
    threads.push_back(thread);
  }

  for (auto thread : threads) { thread->join(); }

  EXPECT_EQ(1, counter->get_map_size());
  EXPECT_TRUE(counter->contains_column({column.data, column.valid}));

  threads.clear();

  for (std::size_t i = 0; i < length; i++) {
    auto thread = std::make_shared<std::thread>(
      [&counter, &column]() { counter->increment(&column); });
    threads.push_back(thread);
  }

  for (auto thread : threads) { thread->join(); }

  EXPECT_EQ(1, counter->get_map_size());
  EXPECT_TRUE(counter->contains_column({column.data, column.valid}));

  threads.clear();

  for (std::size_t i = 0; i < length; i++) {
    auto thread = std::make_shared<std::thread>(
      [&counter, &column]() { counter->decrement(&column); });
    threads.push_back(thread);
  }

  for (auto thread : threads) {
    thread->join();
    EXPECT_EQ(1, counter->get_map_size());
    EXPECT_TRUE(counter->contains_column({column.data, column.valid}));
  }

  counter->decrement(&column);
  EXPECT_EQ(0, counter->get_map_size());
  EXPECT_FALSE(counter->contains_column({column.data, column.valid}));
}
