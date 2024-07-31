#ifndef SQUINT_CORE_ERROR_CHECKING_HPP
#define SQUINT_CORE_ERROR_CHECKING_HPP

namespace squint {

// Template parameter to specify the error checking policy for tensors and quantities
enum class error_checking { enabled, disabled };

// Helper struct to determine the resulting error checking policy when combining two error checking policies in an expression
template <error_checking ErrorChecking1, error_checking ErrorChecking2> struct resulting_error_checking {
    static constexpr auto value = ErrorChecking1 == error_checking::enabled || ErrorChecking2 == error_checking::enabled
                                      ? error_checking::enabled
                                      : error_checking::disabled;
};

} // namespace squint

#endif // SQUINT_CORE_ERROR_CHECKING_HPP