#include <iostream>
#include "gtest/gtest.h"

#include <ctime>

GTEST_API_ int main(int argc, char** argv) {

    std::cout << "Running main() from " << __FILE__ << std::endl;

    std::string test = std::string(std::string(__FILE__) + ", " + std::string(__FUNCTION__) + ", " + std::to_string(__LINE__));
    std::cout << test << std::endl;

    srand (static_cast <unsigned> (time(0)));

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
