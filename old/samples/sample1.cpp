#include <iostream>
#include <typeinfo>
#include <cassert>
#include <vector>
#include "squint/linalg.hpp"


int main(){
    using position3d = squint::tensor<squint::quantities::length, 3>;
    using time = squint::quantities::time;
    using velocity = squint::quantities::velocity;


    const time dt = time::seconds(0.1);
    std::vector<position3d> points{{1, 2, 3}};
    for (auto &p : points)
    {
        std::cout << p << "\n";
        const auto v = p / dt;
        for (const velocity& vi : v)
        {
            std::cout << vi.as_fps() << "\n";
        }
    }
    return 0;
}