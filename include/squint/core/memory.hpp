#ifndef SQUINT_CORE_MEMORY_HPP
#define SQUINT_CORE_MEMORY_HPP

#ifdef _MSC_VER
#define NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

#include <cstdint>

namespace squint {

/**
 * @brief Enumeration to specify the memory space of tensors.
 *
 * This enum class is used as a template parameter to control where tensor data
 * is stored. It affects the performance of certain operations and the compatibility
 * with external libraries.
 */
enum class memory_space : uint8_t { host, device };

/**
 * @brief Enumeration to specify the ownership of tensors.
 *
 * This enum class is used as a template parameter to control the ownership of
 * tensor data.
 *
 *  - owner: the tensor owns the data and is responsible for its deallocation.
 *  - reference: the tensor does not own the data and should not deallocate it.
 */
enum class ownership_type : uint8_t { owner, reference };

} // namespace squint

#endif // SQUINT_CORE_MEMORY_HPP