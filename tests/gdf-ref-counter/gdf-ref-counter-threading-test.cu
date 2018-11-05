#include <thread>
#include <vector>

#include "gdf-ref-counter-test.h"

const std::size_t length = 200;
/*
template <class Callable>
std::vector<std::shared_ptr<std::thread> > MakeThreads(Callable &&callback) {
  std::vector<std::shared_ptr<std::thread> > threads;
  threads.reserve(length);
  for (std::size_t i = 0; i < length; i++) {
    auto thread = std::make_shared<std::thread>(callback);
    threads.push_back(thread);
  }
  return threads;
}

template <class Callable>
void Join(Callable &&callback) {
  for (auto thread : MakeThreads(callback)) { thread->join(); }
}

template <class Callable>
void Detach(Callable &&callback) {
  for (auto thread : MakeThreads(callback)) { thread->detach(); }
}

TEST(GdfRefCounterTest, DISABLE_ThreadingWithJoin) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();
  gdf_column   *  column;

  Join([&counter, column]() { counter->register_column(column); });

  EXPECT_EQ(1, counter->get_map_size());
  EXPECT_TRUE(counter->contains_column({column}));

  Join([&counter, column]() { counter->increment(column); });

  EXPECT_EQ(1, counter->get_map_size());
  EXPECT_TRUE(counter->contains_column({column}));

  auto threads =
    MakeThreads([&counter, column]() { counter->decrement(column); });

  for (auto thread : threads) {
    thread->join();
    EXPECT_EQ(1, counter->get_map_size());
    EXPECT_TRUE(counter->contains_column({column}));
  }

  counter->decrement(column);
  EXPECT_EQ(0, counter->get_map_size());
  EXPECT_FALSE(counter->contains_column({column}));
}

TEST(GdfRefCounterTest, DISABLE_ThreadingWithDetach) {
  GDFRefCounter *counter = GDFRefCounter::getInstance();
  gdf_column     column;

  Detach([&counter, column]() { counter->register_column(column); });

  EXPECT_EQ(1, counter->get_map_size());
  EXPECT_TRUE(counter->contains_column({column}));

  Detach([&counter, column]() { counter->increment(column); });

  EXPECT_EQ(1, counter->get_map_size());
  EXPECT_TRUE(counter->contains_column({column}));

  auto threads =
    MakeThreads([&counter, column]() { counter->decrement(column); });

  for (auto thread : threads) {
    thread->detach();
    EXPECT_EQ(1, counter->get_map_size());
    EXPECT_TRUE(counter->contains_column({column}));
  }

  counter->decrement(column);
  EXPECT_EQ(0, counter->get_map_size());
  EXPECT_FALSE(counter->contains_column({column}));

  delete column;
}
*/
