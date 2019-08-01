#include <blazing/metrics/chronometer.hpp>

#include "StubWatch.hpp"

#include <gtest/gtest.h>

TEST(ChronometerCreationTest, CreateUnstarted) {
    using blazing::metrics::Chronometer;
    auto chronometer = Chronometer::MakeUnstarted();
    EXPECT_FALSE(chronometer->IsRunning());
    EXPECT_EQ(0, chronometer->Elapsed());
}

TEST(ChronometerCreationTest, CreateStarted) {
    using blazing::metrics::Chronometer;
    auto chronometer = Chronometer::MakeStarted();
    EXPECT_TRUE(chronometer->IsRunning());
}

class ChronometerTest : public testing::Test {
protected:
    explicit ChronometerTest()
        : watch_{blazing::metrics::testing::StubWatch::Make()},
          chronometer_{blazing::metrics::Chronometer::Content(*watch_)} {}

    blazing::metrics::testing::StubWatch & watch() const noexcept {
        return *watch_;
    }

    blazing::metrics::Chronometer & chronometer() const noexcept {
        return *chronometer_;
    }

private:
    std::unique_ptr<blazing::metrics::testing::StubWatch> watch_;
    std::unique_ptr<blazing::metrics::Chronometer>        chronometer_;
};


TEST_F(ChronometerTest, Start) {
    EXPECT_FALSE(chronometer().IsRunning());
    EXPECT_EQ(0, chronometer().Elapsed());
}

TEST_F(ChronometerTest, StartAgain) {
    chronometer().Start();
    try {
        chronometer().Start();
        FAIL();
    } catch (const blazing::metrics::IllegalStateException &) {}
    EXPECT_TRUE(chronometer().IsRunning());
}

TEST_F(ChronometerTest, Stop) {
    chronometer().Start();
    EXPECT_EQ(&chronometer(), &chronometer().Stop());
    EXPECT_FALSE(chronometer().IsRunning());
}

TEST_F(ChronometerTest, StopCreated) {
    try {
        chronometer().Stop();
        FAIL();
    } catch (const blazing::metrics::IllegalStateException &) {}
}

TEST_F(ChronometerTest, StopAlreadyStopped) {
    chronometer().Start();
    chronometer().Stop();
    try {
        chronometer().Stop();
        FAIL();
    } catch (const blazing::metrics::IllegalStateException &) {}
    EXPECT_FALSE(chronometer().IsRunning());
}

TEST_F(ChronometerTest, TimelineStart) {
    watch().Add(3);
    chronometer().Reset();
    EXPECT_FALSE(chronometer().IsRunning());
    watch().Add(4);
    EXPECT_EQ(0, chronometer().Elapsed());
    chronometer().Start();
    watch().Add(5);
    EXPECT_EQ(5, chronometer().Elapsed());
}
TEST_F(ChronometerTest, TimelineAgain) {
    watch().Add(3);
    chronometer().Start();
    EXPECT_EQ(0, chronometer().Elapsed());
    watch().Add(4);
    EXPECT_EQ(4, chronometer().Elapsed());
    chronometer().Reset();
    EXPECT_FALSE(chronometer().IsRunning());
    watch().Add(5);
    EXPECT_EQ(0, chronometer().Elapsed());
}

TEST_F(ChronometerTest, TimelineElapsed) {
    watch().Add(9);
    chronometer().Start();
    EXPECT_EQ(0, chronometer().Elapsed());
    watch().Add(7);
    EXPECT_EQ(7, chronometer().Elapsed());
}

TEST_F(ChronometerTest, TimelineNotRunning) {
    watch().Add(5);
    chronometer().Start();
    watch().Add(6);
    chronometer().Stop();
    watch().Add(7);
    EXPECT_EQ(6, chronometer().Elapsed());
}

TEST_F(ChronometerTest, TimelineReuse) {
    chronometer().Start();
    watch().Add(5);
    chronometer().Stop();
    watch().Add(6);
    chronometer().Start();
    EXPECT_EQ(5, chronometer().Elapsed());
    watch().Add(7);
    EXPECT_EQ(12, chronometer().Elapsed());
    chronometer().Stop();
    watch().Add(8);
    EXPECT_EQ(12, chronometer().Elapsed());
}

// TEST_F(ChronometerTest, Milliseconds) {
// chronometer().Start();
// watch().Add(999999);
// EXPECT_EQ(0, chronometer().Elapsed(MILLISECONDS));
// watch().Add(1);
// EXPECT_EQ(1, chronometer().Elapsed(MILLISECONDS));
//}
