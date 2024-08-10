#ifndef SQUINT_CORE_MEMORY_HPP
#define SQUINT_CORE_MEMORY_HPP

#ifdef _MSC_VER
#define NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

#include <cstddef>

namespace squint {

enum class memory_space { host, device };

enum class ownership_type { owner, reference };

} // namespace squint

#endif // SQUINT_CORE_MEMORY_HPP