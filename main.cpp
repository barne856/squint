#include "squint/core/layout.hpp"
#include "squint/squint.hpp"

using namespace squint;

auto main() -> int {
    using s = strides::row_major<shape<3, 2>>;
    using t = strides::column_major<shape<3, 2>>;
}