#include <insight.h>
#include <iostream>

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "No arguments given." << std::endl;
        return 0;
    }    

    Insight insight;
    
    int val = std::stoi(argv[1]);
    
    std::cout << "Square of " << val << " is " << insight.square(val) << std::endl;

    return 0;
}
