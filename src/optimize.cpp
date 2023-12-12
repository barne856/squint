module;
#include <ostream>
#include <iostream>
export module squint:optimize;

import :quantity;
import :tensor;

export namespace squint {
// concepts for functions
template <class F, typename T>
concept single_variable = scalar<T> && std::invocable<F, T> && scalar<typename std::invoke_result<F, T>::type>;
template <class F, typename T>
concept scalar_valued = tensorial<T> && std::invocable<F, T> && scalar<typename std::invoke_result<F, T>::type>;
template <class F, typename T>
concept vector_valued = tensorial<T> && std::invocable<F, T> && tensorial<typename std::invoke_result<F, T>::type>;
// Gradient
template <tensorial T, typename F>
requires(scalar_valued<F, T>) auto grad(const F &func, const T &args) {
    // auto eps = maximum(norm(args) * 1e-8, typename T::value_type(1e-16));
    auto eps = norm(args);
    using res_type = decltype(func(T{}) / std::declval<typename T::value_type &>());
    auto x = args;
    auto g = args.template copy_as<res_type>();
    int i = 0;
    for (auto &elem : x) {
        // central difference
        elem -= eps;
        auto f0 = func(x);
        elem += 2 * eps;
        auto f1 = func(x);
        g[i] = (f1 - f0) / (2.0*eps);
        elem -= eps;
        i++;
    }
    return g;
}
// Jacobian
template <tensorial T, typename F>
requires(vector_valued<F, T>) auto jac(const F &func, const T &args) {
    for (int i = 0; i < args.size(); i++) {
        auto f = [func, i](const T &x) { return func(x)[i].copy(); };
        auto row = grad(f, args).transpose();
        std::cout << row << std::endl;
    }
}

} // namespace squint
