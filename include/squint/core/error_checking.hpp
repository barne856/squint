/**
 * @file error_checking.hpp
 * @brief Defines error checking policies for tensors and quantities.
 *
 * This file provides an enumeration for specifying error checking policies
 * and a helper struct for determining the resulting policy when combining
 * two policies in an expression.
 */

#ifndef SQUINT_CORE_ERROR_CHECKING_HPP
#define SQUINT_CORE_ERROR_CHECKING_HPP

namespace squint {

/**
 * @brief Enumeration to specify the error checking policy for tensors and quantities.
 *
 * This enum class is used as a template parameter to control whether error
 * checking is enabled or disabled for tensor and quantity operations.
 */
enum class error_checking {
    enabled, /**< Error checking is enabled */
    disabled /**< Error checking is disabled */
};

/**
 * @brief Helper struct to determine the resulting error checking policy.
 *
 * This struct is used to determine the resulting error checking policy
 * when combining two error checking policies in an expression. The resulting
 * policy is enabled if at least one of the input policies is enabled.
 *
 * @tparam ErrorChecking1 The first error checking policy.
 * @tparam ErrorChecking2 The second error checking policy.
 */
template <error_checking ErrorChecking1, error_checking ErrorChecking2> struct resulting_error_checking {
    /**
     * @brief The resulting error checking policy.
     *
     * This static constexpr member holds the resulting error checking policy.
     * It is set to enabled if either ErrorChecking1 or ErrorChecking2 is enabled,
     * and disabled otherwise.
     */
    static constexpr auto value = ErrorChecking1 == error_checking::enabled || ErrorChecking2 == error_checking::enabled
                                      ? error_checking::enabled
                                      : error_checking::disabled;
};

} // namespace squint

#endif // SQUINT_CORE_ERROR_CHECKING_HPP