#ifndef SQUINT_CORE_MEMORY_HPP
#define SQUINT_CORE_MEMORY_HPP

#ifdef _MSC_VER
#define NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

#include <cstdint>

namespace squint {

enum class memory_space : uint8_t { host, device };

enum class ownership_type : uint8_t { owner, reference };

} // namespace squint

#endif // SQUINT_CORE_MEMORY_HPP