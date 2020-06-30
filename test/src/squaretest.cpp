#include "gtest/gtest.h"
#include "insight.h"

// Tests square of negative numbers.
TEST(SquareTest, Negative) {
    Insight insight;

    EXPECT_EQ(25, insight.square(-5));
    EXPECT_EQ(1, insight.square(-1));
    EXPECT_GT(insight.square(-10), 0);
}

// Tests square of 0.
TEST(SquareTest, Zero) {
    Insight insight;

    EXPECT_EQ(0, insight.square(0));
}


// Tests square of positive numbers.
TEST(SquareTest, Positive) {
    Insight insight;

    EXPECT_EQ(1, insight.square(1));
    EXPECT_EQ(4, insight.square(2));
    EXPECT_EQ(9, insight.square(3));
    EXPECT_EQ(64, insight.square(8));
}

