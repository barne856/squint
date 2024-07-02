/**
 * @file quantity.hpp
 * @author Brendan Barnes
 * @brief Compile-time quantity implementation
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef SQUINT_QUANTITY_HPP
#define SQUINT_QUANTITY_HPP
#define _USE_MATH_DEFINES
#include <cassert>
#include <cmath>
#include <ostream>
#include "squint/dimension.hpp"
#include "squint/tensor.hpp"

// if compiler is MSVC, define math constants
#ifdef _MSC_VER
#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
#define M_PI_4 0.785398163397448309616
#endif



namespace squint {
namespace quantities {
template <typename T, dimensional D> class quantity {
  public:
    using value_type = T;     // data type of the element
    using dimension_type = D; // dimension of the quantity
    constexpr quantity() = default;
    // implicit conversion from a scalar of type quantity
    template <scalar S>
        requires(tensorial<S> && std::is_same<quantity, typename S::value_type>::value)
    constexpr quantity(const S &other) : _elem(other.data()[0]._elem) {}
    constexpr quantity(const quantity &other) : _elem(other._elem) {}
    // implicit conversion from value type
    constexpr quantity(const T &other) : _elem(other) {}
    operator const T &() const
        requires(std::is_same<D, dimensions::dimensionless>::value)
    {
        return _elem;
    }
    // explicit conversion to pointer to value
    explicit operator T *() { return &_elem; }
    explicit operator const T *() const { return &_elem; }
    // explicit conversion to another dimension
    template <dimensional P> explicit operator quantity<T, P>() const { return quantity<T, P>(_elem); }
    // three way comparison
    auto operator<=>(const quantity &rhs) const { return _elem <=> rhs._elem; }
    bool operator==(const quantity &rhs) const { return _elem == rhs._elem; }
    // math operators
    quantity operator-() const { return quantity(-_elem); }
    quantity &operator+=(const quantity &rhs) {
        _elem += rhs._elem;
        return *this;
    }
    quantity &operator-=(const quantity &rhs) {
        _elem -= rhs._elem;
        return *this;
    }
    quantity &operator*=(const T &rhs) {
        _elem *= rhs;
        return *this;
    }
    quantity &operator/=(const T &rhs) {
        _elem /= rhs;
        return *this;
    }
    quantity operator+(const quantity &rhs) const { return quantity(_elem + rhs._elem); }
    quantity operator-(const quantity &rhs) const { return quantity(_elem - rhs._elem); }

    T _elem{};

    // cast to quantity from underlying type
    static constexpr quantity from_value(T d) { return quantity(d); }
    // lengths ---------------------------------------------------------------------------------------------------------
    static constexpr quantity feet(T d)
        requires(std::is_same<D, dimensions::length>::value)
    {
        return quantity(d * 0.3048);
    }
    static constexpr quantity yards(T d)
        requires(std::is_same<D, dimensions::length>::value)
    {
        return quantity(d * 0.3048 * 3.);
    }
    static constexpr quantity inches(T d)
        requires(std::is_same<D, dimensions::length>::value)
    {
        return quantity(d * 0.3048 / 12.);
    }
    static constexpr quantity meters(T d)
        requires(std::is_same<D, dimensions::length>::value)
    {
        return quantity(d);
    }
    static constexpr quantity miles(T d)
        requires(std::is_same<D, dimensions::length>::value)
    {
        return quantity(d * 5280. * 0.3048);
    }
    T as_feet() const
        requires(std::is_same<D, dimensions::length>::value)
    {
        return _elem / 0.3048;
    }
    T as_yards() const
        requires(std::is_same<D, dimensions::length>::value)
    {
        return _elem / 0.3048 / 3.;
    }
    T as_inches() const
        requires(std::is_same<D, dimensions::length>::value)
    {
        return _elem / 0.3048 * 12.;
    }
    T as_meters() const
        requires(std::is_same<D, dimensions::length>::value)
    {
        return _elem;
    }
    T as_miles() const
        requires(std::is_same<D, dimensions::length>::value)
    {
        return _elem / 5280. / 0.3048;
    }
    // area ------------------------------------------------------------------------------------------------------------
    static constexpr quantity square_feet(T d)
        requires(std::is_same<D, dimensions::area>::value)
    {
        return quantity(d * 0.3048 * 0.3048);
    }
    static constexpr quantity square_yards(T d)
        requires(std::is_same<D, dimensions::area>::value)
    {
        return quantity(d * 0.3048 * 0.3048 * 3. * 3.);
    }
    static constexpr quantity square_inches(T d)
        requires(std::is_same<D, dimensions::area>::value)
    {
        return quantity(d * 0.3048 * 0.3048 / 12. / 12.);
    }
    static constexpr quantity square_meters(T d)
        requires(std::is_same<D, dimensions::area>::value)
    {
        return quantity(d);
    }
    static constexpr quantity square_miles(T d)
        requires(std::is_same<D, dimensions::area>::value)
    {
        return quantity(d * 5280. * 0.3048 * 5280. * 0.3048);
    }
    static constexpr quantity acres(T d)
        requires(std::is_same<D, dimensions::area>::value)
    {
        return quantity(d * 4046.8564224);
    }
    T as_square_feet() const
        requires(std::is_same<D, dimensions::area>::value)
    {
        return _elem / 0.3048 / 0.3048;
    }
    T as_square_yards() const
        requires(std::is_same<D, dimensions::area>::value)
    {
        return _elem / 0.3048 / 3. / 0.3048 / 3.;
    }
    T as_square_inches() const
        requires(std::is_same<D, dimensions::area>::value)
    {
        return _elem / 0.3048 / 0.3048 * 12. * 12.;
    }
    T as_square_meters() const
        requires(std::is_same<D, dimensions::area>::value)
    {
        return _elem;
    }
    T as_square_miles() const
        requires(std::is_same<D, dimensions::area>::value)
    {
        return _elem / 5280. / 0.3048 / 5280. / 0.3048;
    }
    T as_acres() const
        requires(std::is_same<D, dimensions::area>::value)
    {
        return _elem / 4046.8564224;
    }
    // time ------------------------------------------------------------------------------------------------------------
    static constexpr quantity seconds(T d)
        requires(std::is_same<D, dimensions::time>::value)
    {
        return quantity(d);
    }
    T as_seconds() const
        requires(std::is_same<D, dimensions::time>::value)
    {
        return _elem;
    }
    // mass ------------------------------------------------------------------------------------------------------------
    static constexpr quantity kilograms(T d)
        requires(std::is_same<D, dimensions::mass>::value)
    {
        return quantity(d);
    }
    static constexpr quantity slugs(T d)
        requires(std::is_same<D, dimensions::mass>::value)
    {
        return quantity(d * 14.5939029372);
    }
    T as_kilograms() const
        requires(std::is_same<D, dimensions::mass>::value)
    {
        return _elem;
    }
    T as_slugs() const
        requires(std::is_same<D, dimensions::mass>::value)
    {
        return _elem / 14.5939029372;
    }
    // temperature -----------------------------------------------------------------------------------------------------
    static constexpr quantity kelvin(T d)
        requires(std::is_same<D, dimensions::temperature>::value)
    {
        return quantity(d);
    }
    static constexpr quantity celcius(T d)
        requires(std::is_same<D, dimensions::temperature>::value)
    {
        return quantity(d + 273.15);
    }
    static constexpr quantity fahrenheit(T d)
        requires(std::is_same<D, dimensions::temperature>::value)
    {
        return quantity(273.15 + (d - 32.) * 5. / 9.);
    }
    T as_kelvin() const
        requires(std::is_same<D, dimensions::temperature>::value)
    {
        return _elem;
    }
    T as_celcius() const
        requires(std::is_same<D, dimensions::temperature>::value)
    {
        return _elem - 273.15;
    }
    T as_fahrenheit() const
        requires(std::is_same<D, dimensions::temperature>::value)
    {
        return 32. + ((_elem - 273.15) * 9. / 5.);
    }
    // volume ----------------------------------------------------------------------------------------------------------
    static constexpr quantity cubic_meters(T d)
        requires(std::is_same<D, dimensions::volume>::value)
    {
        return quantity(d);
    }
    static constexpr quantity cubic_feet(T d)
        requires(std::is_same<D, dimensions::volume>::value)
    {
        return quantity(d * 0.02831685);
    }
    static constexpr quantity liters(T d)
        requires(std::is_same<D, dimensions::volume>::value)
    {
        return quantity(d * 0.001);
    }
    static constexpr quantity gallons(T d)
        requires(std::is_same<D, dimensions::volume>::value)
    {
        return quantity(d * 0.00378541);
    }
    T as_cubic_meters() const
        requires(std::is_same<D, dimensions::volume>::value)
    {
        return _elem;
    }
    T as_cubic_feet() const
        requires(std::is_same<D, dimensions::volume>::value)
    {
        return _elem / 0.02831685;
    }
    T as_liters() const
        requires(std::is_same<D, dimensions::volume>::value)
    {
        return _elem / 0.001;
    }
    T as_gallons() const
        requires(std::is_same<D, dimensions::volume>::value)
    {
        return _elem / 0.00378541;
    }
    // flow ------------------------------------------------------------------------------------------------------------
    static constexpr quantity cms(T d)
        requires(std::is_same<D, dimensions::flow>::value)
    {
        return quantity(d);
    }
    static constexpr quantity cfs(T d)
        requires(std::is_same<D, dimensions::flow>::value)
    {
        return quantity(d * 0.028316847);
    }
    static constexpr quantity gpm(T d)
        requires(std::is_same<D, dimensions::flow>::value)
    {
        return quantity(d * 15850.3);
    }
    static constexpr quantity gpd(T d)
        requires(std::is_same<D, dimensions::flow>::value)
    {
        return quantity(d * 15850.3 * 60. * 24.);
    }
    static constexpr quantity mgd(T d)
        requires(std::is_same<D, dimensions::flow>::value)
    {
        return quantity(d * 15850.3 * 60. * 24. / 1000000.);
    }
    T as_cms() const
        requires(std::is_same<D, dimensions::flow>::value)
    {
        return _elem;
    }
    T as_cfs() const
        requires(std::is_same<D, dimensions::flow>::value)
    {
        return _elem / 0.02831685;
    }
    T as_gpm() const
        requires(std::is_same<D, dimensions::flow>::value)
    {
        return _elem / 15850.3;
    }
    T as_gpd() const
        requires(std::is_same<D, dimensions::flow>::value)
    {
        return _elem / (15850.3 * 60. * 24.);
    }
    T as_mgd() const
        requires(std::is_same<D, dimensions::flow>::value)
    {
        return 1000000. * _elem / (15850.3 * 60. * 24.);
    }
    // velocity --------------------------------------------------------------------------------------------------------
    static constexpr quantity mps(T d)
        requires(std::is_same<D, dimensions::velocity>::value)
    {
        return quantity(d);
    }
    static constexpr quantity fps(T d)
        requires(std::is_same<D, dimensions::velocity>::value)
    {
        return quantity(d * 0.3048);
    }
    T as_mps() const
        requires(std::is_same<D, dimensions::velocity>::value)
    {
        return _elem;
    }
    T as_fps() const
        requires(std::is_same<D, dimensions::velocity>::value)
    {
        return _elem / 0.3048;
    }
    // acceleration ----------------------------------------------------------------------------------------------------
    static constexpr quantity mps2(T d)
        requires(std::is_same<D, dimensions::acceleration>::value)
    {
        return quantity(d);
    }
    static constexpr quantity fps2(T d)
        requires(std::is_same<D, dimensions::acceleration>::value)
    {
        return quantity(d * 0.3048);
    }
    T as_mps2() const
        requires(std::is_same<D, dimensions::acceleration>::value)
    {
        return _elem;
    }
    T as_fps2() const
        requires(std::is_same<D, dimensions::acceleration>::value)
    {
        return _elem / 0.3048;
    }
    // density ---------------------------------------------------------------------------------------------------------
    static constexpr quantity kilograms_per_cubic_meter(T d)
        requires(std::is_same<D, dimensions::density>::value)
    {
        return quantity(d);
    }
    static constexpr quantity slugs_per_cubic_foot(T d)
        requires(std::is_same<D, dimensions::density>::value)
    {
        return quantity(d * 515.3788184);
    }
    T as_kilograms_per_cubic_meter() const
        requires(std::is_same<D, dimensions::density>::value)
    {
        return _elem;
    }
    T as_slugs_per_cubic_foot() const
        requires(std::is_same<D, dimensions::density>::value)
    {
        return _elem / 515.3788184;
    }
    // dynamic viscosity -----------------------------------------------------------------------------------------------
    static constexpr quantity pascal_seconds(T d)
        requires(std::is_same<D, dimensions::dynamic_viscosity>::value)
    {
        return quantity(d);
    }
    static constexpr quantity psf_seconds(T d)
        requires(std::is_same<D, dimensions::dynamic_viscosity>::value)
    {
        return quantity(d * 47.880208);
    }
    T as_pascal_seconds() const
        requires(std::is_same<D, dimensions::dynamic_viscosity>::value)
    {
        return _elem;
    }
    T as_psf_seconds() const
        requires(std::is_same<D, dimensions::dynamic_viscosity>::value)
    {
        return _elem / 515.3788184;
    }
    // kinematic viscosity
    // -----------------------------------------------------------------------------------------------
    static constexpr quantity square_meters_per_second(T d)
        requires(std::is_same<D, dimensions::kinematic_viscosity>::value)
    {
        return quantity(d);
    }
    static constexpr quantity square_feet_per_second(T d)
        requires(std::is_same<D, dimensions::kinematic_viscosity>::value)
    {
        return quantity(d * 0.3048 * 0.3048);
    }
    T as_square_meters_per_second() const
        requires(std::is_same<D, dimensions::kinematic_viscosity>::value)
    {
        return _elem;
    }
    T as_square_feet_per_second() const
        requires(std::is_same<D, dimensions::kinematic_viscosity>::value)
    {
        return _elem / 0.3048 / 0.3048;
    }
    // force -----------------------------------------------------------------------------------------------------------
    static constexpr quantity newtons(T d)
        requires(std::is_same<D, dimensions::force>::value)
    {
        return quantity(d);
    }
    static constexpr quantity pounds(T d)
        requires(std::is_same<D, dimensions::force>::value)
    {
        return quantity(d * 4.4482216);
    }
    T as_newtons() const
        requires(std::is_same<D, dimensions::force>::value)
    {
        return _elem;
    }
    T as_pounds() const
        requires(std::is_same<D, dimensions::force>::value)
    {
        return _elem / 4.4482216;
    }
    // force density ---------------------------------------------------------------------------------------------------
    static constexpr quantity newtons_per_cubic_meter(T d)
        requires(std::is_same<D, dimensions::force_density>::value)
    {
        return quantity(d);
    }
    static constexpr quantity pcf(T d)
        requires(std::is_same<D, dimensions::force_density>::value)
    {
        return quantity(d * 157.0865730619);
    }
    T as_newtons_per_cubic_meter() const
        requires(std::is_same<D, dimensions::force_density>::value)
    {
        return _elem;
    }
    T as_pcf() const
        requires(std::is_same<D, dimensions::force_density>::value)
    {
        return _elem / 157.0865730619;
    }
    // dimensionless ---------------------------------------------------------------------------------------------------
    static constexpr quantity radians(T d)
        requires(std::is_same<D, dimensions::dimensionless>::value)
    {
        return quantity(d);
    }
    static constexpr quantity degrees(T d)
        requires(std::is_same<D, dimensions::dimensionless>::value)
    {
        return quantity(d * M_PI / 180.);
    }
    T as_radians() const
        requires(std::is_same<D, dimensions::force_density>::value)
    {
        return _elem;
    }
    T as_degrees() const
        requires(std::is_same<D, dimensions::force_density>::value)
    {
        return _elem * 180. / M_PI;
    }
    // pressure --------------------------------------------------------------------------------------------------------
    static constexpr quantity pascals(T d)
        requires(std::is_same<D, dimensions::pressure>::value)
    {
        return quantity(d);
    }
    static constexpr quantity psf(T d)
        requires(std::is_same<D, dimensions::pressure>::value)
    {
        return quantity(d * 47.880258888889);
    }
    static constexpr quantity psi(T d)
        requires(std::is_same<D, dimensions::pressure>::value)
    {
        return quantity(d * 6894.7572932);
    }
    T as_pascals() const
        requires(std::is_same<D, dimensions::force_density>::value)
    {
        return _elem;
    }
    T as_psf() const
        requires(std::is_same<D, dimensions::force_density>::value)
    {
        return _elem / 47.880258888889;
    }
    T as_psi() const
        requires(std::is_same<D, dimensions::force_density>::value)
    {
        return _elem / 6894.7572932;
    }
};

// / and * create new types
template <quantitative Q1, quantitative Q2>
using quant_mult_t =
    quantity<decltype(std::declval<typename Q1::value_type &>() * std::declval<typename Q2::value_type &>()),
             mult_t<typename Q1::dimension_type, typename Q2::dimension_type>>;
template <quantitative Q1, quantitative Q2>
using quant_div_t =
    quantity<decltype(std::declval<typename Q1::value_type &>() / std::declval<typename Q2::value_type &>()),
             div_t<typename Q1::dimension_type, typename Q2::dimension_type>>;

// multiply and divide quantities
template <quantitative Q1, quantitative Q2> quant_mult_t<Q1, Q2> operator*(const Q1 &lhs, const Q2 &rhs) {
    return quant_mult_t<Q1, Q2>(lhs._elem * rhs._elem);
}
template <quantitative Q1, quantitative Q2> quant_div_t<Q1, Q2> operator/(const Q1 &lhs, const Q2 &rhs) {
    return quant_div_t<Q1, Q2>(lhs._elem / rhs._elem);
}

// multiply and divide by value type
template <quantitative Q> auto operator*(const Q &lhs, const typename Q::value_type &rhs) {
    return lhs * quantity<typename Q::value_type, dimensions::dimensionless>(rhs);
}
template <quantitative Q> auto operator/(const Q &lhs, const typename Q::value_type &rhs) {
    return lhs / quantity<typename Q::value_type, dimensions::dimensionless>(rhs);
}
template <quantitative Q> auto operator*(const typename Q::value_type &lhs, const Q &rhs) {
    return quantity<typename Q::value_type, dimensions::dimensionless>(lhs) * rhs;
}
template <quantitative Q> auto operator/(const typename Q::value_type &lhs, const Q &rhs) {
    return quantity<typename Q::value_type, dimensions::dimensionless>(lhs) / rhs;
}

// multiply and divide by tensor of scalar shape
template <quantitative Q, scalar S>
    requires(tensor_shape<S>)
auto operator*(const Q &lhs, const S &rhs) {
    return lhs * rhs.data()[0];
}
template <quantitative Q, scalar S>
    requires(tensor_shape<S>)
auto operator/(const Q &lhs, const S &rhs) {
    return lhs / rhs.data()[0];
}
template <quantitative Q, scalar S>
    requires(tensor_shape<S>)
auto operator*(const S &lhs, const Q &rhs) {
    return lhs.data()[0] * rhs;
}
template <quantitative Q, scalar S>
    requires(tensor_shape<S>)
auto operator/(const S &lhs, const Q &rhs) {
    return lhs.data()[0] / rhs;
}

// divide by dynamic tensor of scalar shape (multiplication not needed since it is handled by tensor operators)
// division by quantity also not needed since it is handled by tensor operators
template <quantitative Q, dynamic_tensor S> auto operator/(const Q &lhs, const S &rhs) {
    assert(rhs.size() == 1);
    return lhs / rhs.data()[0];
}

// add and subtract by tensor of scalar shape
template <quantitative Q, scalar S>
    requires(tensor_shape<S> && std::is_same<typename S::value_type, Q>::value)
auto operator+(const Q &lhs, const S &rhs) {
    return lhs + rhs.data()[0];
}
template <quantitative Q, scalar S>
    requires(tensor_shape<S> && std::is_same<typename S::value_type, Q>::value)
auto operator-(const Q &lhs, const S &rhs) {
    return lhs - rhs.data()[0];
}
template <quantitative Q, scalar S>
    requires(tensor_shape<S> && std::is_same<typename S::value_type, Q>::value)
auto operator+(const S &lhs, const Q &rhs) {
    return lhs.data()[0] + rhs;
}
template <quantitative Q, scalar S>
    requires(tensor_shape<S> && std::is_same<typename S::value_type, Q>::value)
auto operator-(const S &lhs, const Q &rhs) {
    return lhs.data()[0] - rhs;
}

// add and subtract by dynamic tensor of scalar shape
template <quantitative Q, dynamic_tensor S>
    requires(std::is_same<typename S::value_type, Q>::value)
auto operator+(const Q &lhs, const S &rhs) {
    return lhs + rhs.data()[0];
}
template <quantitative Q, dynamic_tensor S>
    requires(std::is_same<typename S::value_type, Q>::value)
auto operator-(const Q &lhs, const S &rhs) {
    return lhs - rhs.data()[0];
}
template <quantitative Q, dynamic_tensor S>
    requires(std::is_same<typename S::value_type, Q>::value)
auto operator+(const S &lhs, const Q &rhs) {
    return lhs.data()[0] + rhs;
}
template <quantitative Q, dynamic_tensor S>
    requires(std::is_same<typename S::value_type, Q>::value)
auto operator-(const S &lhs, const Q &rhs) {
    return lhs.data()[0] - rhs;
}

// comparisons with dimensionless quantities
template <quantitative Q>
    requires(std::is_same<typename Q::dimension_type, dimensions::dimensionless>::value)
bool operator<=>(const Q &lhs, const typename Q::value_type &rhs) {
    return lhs._elem <=> rhs;
}
template <quantitative Q>
    requires(std::is_same<typename Q::dimension_type, dimensions::dimensionless>::value)
bool operator==(const Q &lhs, const typename Q::value_type &rhs) {
    return lhs._elem == rhs;
}
template <quantitative Q>
    requires(std::is_same<typename Q::dimension_type, dimensions::dimensionless>::value)
bool operator<=>(const typename Q::value_type &lhs, const Q &rhs) {
    return lhs._elem <=> rhs;
}
template <quantitative Q>
    requires(std::is_same<typename Q::dimension_type, dimensions::dimensionless>::value)
bool operator==(const typename Q::value_type &lhs, const Q &rhs) {
    return lhs._elem == rhs;
}
// printing
template <quantitative Q> std::ostream &operator<<(std::ostream &os, const Q &quant) {
    os << quant._elem;
    return os;
}

// Unit Literals -------------------------------------------------------------------------------------------------------
// Length
using length = quantity<double, dimensions::length>;
using length_f = quantity<float, dimensions::length>;
// Meters
constexpr length operator""_m(long double d) { return length::meters(static_cast<double>(d)); }
constexpr length operator""_m(unsigned long long int d) { return length::meters(static_cast<double>(d)); }
// Yard
constexpr length operator""_yd(long double d) { return length::yards(static_cast<double>(d)); }
constexpr length operator""_yd(unsigned long long int d) { return length::yards(static_cast<double>(d)); }
// Feet
constexpr length operator""_ft(long double d) { return length::feet(static_cast<double>(d)); }
constexpr length operator""_ft(unsigned long long int d) { return length::feet(static_cast<double>(d)); }
// Inches
constexpr length operator""_in(long double d) { return length::inches(static_cast<double>(d)); }
constexpr length operator""_in(unsigned long long int d) { return length::inches(static_cast<double>(d)); }
// Miles
constexpr length operator""_mi(long double d) { return length::miles(static_cast<double>(d)); }
constexpr length operator""_mi(unsigned long long int d) { return length::miles(static_cast<double>(d)); }
// Area
using area = quantity<double, dimensions::area>;
using area_f = quantity<float, dimensions::area>;
// Square Meters
constexpr area operator""_sqm(long double d) { return area::square_meters(static_cast<double>(d)); }
constexpr area operator""_sqm(unsigned long long int d) { return area::square_meters(static_cast<double>(d)); }
// Square Yard
constexpr area operator""_sqyd(long double d) { return area::square_yards(static_cast<double>(d)); }
constexpr area operator""_sqyd(unsigned long long int d) { return area::square_yards(static_cast<double>(d)); }
// Square Feet
constexpr area operator""_sqft(long double d) { return area::square_feet(static_cast<double>(d)); }
constexpr area operator""_sqft(unsigned long long int d) { return area::square_feet(static_cast<double>(d)); }
// Square Inches
constexpr area operator""_sqin(long double d) { return area::square_inches(static_cast<double>(d)); }
constexpr area operator""_sqin(unsigned long long int d) { return area::square_inches(static_cast<double>(d)); }
// Square Miles
constexpr area operator""_sqmi(long double d) { return area::square_miles(static_cast<double>(d)); }
constexpr area operator""_sqmi(unsigned long long int d) { return area::square_miles(static_cast<double>(d)); }
// Acres
constexpr area operator""_ac(long double d) { return area::acres(static_cast<double>(d)); }
constexpr area operator""_ac(unsigned long long int d) { return area::acres(static_cast<double>(d)); }
// Time
using time = quantity<double, dimensions::time>;
using time_f = quantity<float, dimensions::time>;
// Seconds
constexpr time operator""_s(long double d) { return time::seconds(static_cast<double>(d)); }
constexpr time operator""_s(unsigned long long int d) { return time::seconds(static_cast<double>(d)); }
// Mass
using mass = quantity<double, dimensions::mass>;
using mass_f = quantity<float, dimensions::mass>;
// Kilograms
constexpr mass operator""_kg(long double d) { return mass::kilograms(static_cast<double>(d)); }
constexpr mass operator""_kg(unsigned long long int d) { return mass::kilograms(static_cast<double>(d)); }
// Slug
constexpr mass operator""_slug(long double d) { return mass::slugs(static_cast<double>(d)); }
constexpr mass operator""_slug(unsigned long long int d) { return mass::slugs(static_cast<double>(d)); }
// Temperature
using temperature = quantity<double, dimensions::temperature>;
using temperature_f = quantity<float, dimensions::temperature>;
// Kelvin
constexpr temperature operator""_K(long double d) { return temperature::kelvin(static_cast<double>(d)); }
constexpr temperature operator""_K(unsigned long long int d) { return temperature::kelvin(static_cast<double>(d)); }
// Celcius
constexpr temperature operator""_C(long double d) { return temperature::celcius(static_cast<double>(d)); }
constexpr temperature operator""_C(unsigned long long int d) { return temperature::celcius(static_cast<double>(d)); }
// Fahrenheit
constexpr temperature operator""_F(long double d) { return temperature::fahrenheit(static_cast<double>(d)); }
constexpr temperature operator""_F(unsigned long long int d) { return temperature::fahrenheit(static_cast<double>(d)); }
// Volume
using volume = quantity<double, dimensions::volume>;
using volume_f = quantity<float, dimensions::volume>;
// Cubic Meters
constexpr volume operator""_m3(long double d) { return volume::cubic_meters(static_cast<double>(d)); }
constexpr volume operator""_m3(unsigned long long int d) { return volume::cubic_meters(static_cast<double>(d)); }
// Cubic feet
constexpr volume operator""_ft3(long double d) { return volume::cubic_feet(static_cast<double>(d)); }
constexpr volume operator""_ft3(unsigned long long int d) { return volume::cubic_feet(static_cast<double>(d)); }
// Liters
constexpr volume operator""_l(long double d) { return volume::liters(static_cast<double>(d)); }
constexpr volume operator""_l(unsigned long long int d) { return volume::liters(static_cast<double>(d)); }
// Gallons
constexpr volume operator""_gal(long double d) { return volume::gallons(static_cast<double>(d)); }
constexpr volume operator""_gal(unsigned long long int d) { return volume::gallons(static_cast<double>(d)); }
// Flow
using flow = quantity<double, dimensions::flow>;
using flow_f = quantity<float, dimensions::flow>;
// Cubic Meters per Second
constexpr flow operator""_cms(long double d) { return flow::cms(static_cast<double>(d)); }
constexpr flow operator""_cms(unsigned long long int d) { return flow::cms(static_cast<double>(d)); }
// Cubic Feet per Second
constexpr flow operator""_cfs(long double d) { return flow::cfs(static_cast<double>(d)); }
constexpr flow operator""_cfs(unsigned long long int d) { return flow::cfs(static_cast<double>(d)); }
// Gallons per Minute
constexpr flow operator""_gpm(long double d) { return flow::gpm(static_cast<double>(d)); }
constexpr flow operator""_gpm(unsigned long long int d) { return flow::gpm(static_cast<double>(d)); }
// Gallons per Day
constexpr flow operator""_gpd(long double d) { return flow::gpd(static_cast<double>(d)); }
constexpr flow operator""_gpd(unsigned long long int d) { return flow::gpd(static_cast<double>(d)); }
// Million Gallons per Day
constexpr flow operator""_mgd(long double d) { return flow::mgd(static_cast<double>(d)); }
constexpr flow operator""_mgd(unsigned long long int d) { return flow::mgd(static_cast<double>(d)); }
// Velocity
using velocity = quantity<double, dimensions::velocity>;
using velocity_f = quantity<float, dimensions::velocity>;
// Meters per Second
constexpr velocity operator""_mps(long double d) { return velocity::mps(static_cast<double>(d)); }
constexpr velocity operator""_mps(unsigned long long int d) { return velocity::mps(static_cast<double>(d)); }
// Feet per Second
constexpr velocity operator""_fps(long double d) { return velocity::fps(static_cast<double>(d)); }
constexpr velocity operator""_fps(unsigned long long int d) { return velocity::fps(static_cast<double>(d)); }
// Acceleration
using acceleration = quantity<double, dimensions::acceleration>;
using acceleration_f = quantity<float, dimensions::acceleration>;
// Meters per Second Squared
constexpr acceleration operator""_mps2(long double d) { return acceleration::mps2(static_cast<double>(d)); }
constexpr acceleration operator""_mps2(unsigned long long int d) { return acceleration::mps2(static_cast<double>(d)); }
// Feet per Second Squared
constexpr acceleration operator""_fps2(long double d) { return acceleration::fps2(static_cast<double>(d)); }
constexpr acceleration operator""_fps2(unsigned long long int d) { return acceleration::fps2(static_cast<double>(d)); }
// Density
using density = quantity<double, dimensions::density>;
using density_f = quantity<float, dimensions::density>;
// Kilograms per Cubic Meter
constexpr density operator""_kgm3(long double d) { return density::kilograms_per_cubic_meter(static_cast<double>(d)); }
constexpr density operator""_kgm3(unsigned long long int d) {
    return density::kilograms_per_cubic_meter(static_cast<double>(d));
}
// Slug per Cubic Feet
constexpr density operator""_slugcf(long double d) { return density::slugs_per_cubic_foot(static_cast<double>(d)); }
constexpr density operator""_slugcf(unsigned long long int d) {
    return density::slugs_per_cubic_foot(static_cast<double>(d));
}
// Dynamic Viscosity
using dynamic_viscosity = quantity<double, dimensions::dynamic_viscosity>;
using dynamic_viscosity_f = quantity<float, dimensions::dynamic_viscosity>;
// Pascal Seconds
constexpr dynamic_viscosity operator""_pas(long double d) {
    return dynamic_viscosity::pascal_seconds(static_cast<double>(d));
}
constexpr dynamic_viscosity operator""_pas(unsigned long long int d) {
    return dynamic_viscosity::pascal_seconds(static_cast<double>(d));
}
// Pounds per Square Foot Second
constexpr dynamic_viscosity operator""_psfs(long double d) {
    return dynamic_viscosity::psf_seconds(static_cast<double>(d));
}
constexpr dynamic_viscosity operator""_psfs(unsigned long long int d) {
    return dynamic_viscosity::psf_seconds(static_cast<double>(d));
}
// Kinematic Viscosity
using kinematic_viscosity = quantity<double, dimensions::kinematic_viscosity>;
using kinematic_viscosity_f = quantity<float, dimensions::kinematic_viscosity>;
// Square Meters per Second
constexpr kinematic_viscosity operator""_m2ps(long double d) {
    return kinematic_viscosity::square_meters_per_second(static_cast<double>(d));
}
constexpr kinematic_viscosity operator""_m2ps(unsigned long long int d) {
    return kinematic_viscosity::square_meters_per_second(static_cast<double>(d));
}
// Square Feet per Second
constexpr kinematic_viscosity operator""_ft2ps(long double d) {
    return kinematic_viscosity::square_feet_per_second(static_cast<double>(d));
}
constexpr kinematic_viscosity operator""_ft2ps(unsigned long long int d) {
    return kinematic_viscosity::square_feet_per_second(static_cast<double>(d));
}
// Force
using force = quantity<double, dimensions::force>;
using force_f = quantity<float, dimensions::force>;
// Newtons
constexpr force operator""_n(long double d) { return force::newtons(static_cast<double>(d)); }
constexpr force operator""_n(unsigned long long int d) { return force::newtons(static_cast<double>(d)); }
// Pounds (Force)
constexpr force operator""_lbf(long double d) { return force::pounds(static_cast<double>(d)); }
constexpr force operator""_lbf(unsigned long long int d) { return force::pounds(static_cast<double>(d)); }
// Force Density
using force_density = quantity<double, dimensions::force_density>;
using force_density_f = quantity<float, dimensions::force_density>;
// Newtons per Cubic Meter
constexpr force_density operator""_ncm(long double d) {
    return force_density::newtons_per_cubic_meter(static_cast<double>(d));
}
constexpr force_density operator""_ncm(unsigned long long int d) {
    return force_density::newtons_per_cubic_meter(static_cast<double>(d));
}
// Pounds (Force) per Cubic Foot
constexpr force_density operator""_pcf(long double d) { return force_density::pcf(static_cast<double>(d)); }
constexpr force_density operator""_pcf(unsigned long long int d) { return force_density::pcf(static_cast<double>(d)); }
// Dimensionless
using dimensionless = quantity<double, dimensions::dimensionless>;
using dimensionless_f = quantity<float, dimensions::dimensionless>;
// Radians
constexpr dimensionless operator""_rads(long double d) { return dimensionless::radians(static_cast<double>(d)); }
constexpr dimensionless operator""_rads(unsigned long long int d) {
    return dimensionless::radians(static_cast<double>(d));
}
// Degrees
constexpr dimensionless operator""_degrees(long double d) { return dimensionless::degrees(static_cast<double>(d)); }
constexpr dimensionless operator""_degrees(unsigned long long int d) {
    return dimensionless::degrees(static_cast<double>(d));
}
// Dimensionless
constexpr dimensionless operator""_pure(long double d) { return dimensionless::radians(static_cast<double>(d)); }
constexpr dimensionless operator""_pure(unsigned long long int d) {
    return dimensionless::radians(static_cast<double>(d));
}
// Pressure
using pressure = quantity<double, dimensions::pressure>;
using pressure_f = quantity<float, dimensions::pressure>;
// Pascal
constexpr pressure operator""_pa(long double d) { return pressure::pascals(static_cast<double>(d)); }
constexpr pressure operator""_pa(unsigned long long int d) { return pressure::pascals(static_cast<double>(d)); }
// Pounds per Square Foot
constexpr pressure operator""_psf(long double d) { return pressure::psf(static_cast<double>(d)); }
constexpr pressure operator""_psf(unsigned long long int d) { return pressure::psf(static_cast<double>(d)); }
// Pounds per Square Inch
constexpr pressure operator""_psi(long double d) { return pressure::psi(static_cast<double>(d)); }
constexpr pressure operator""_psi(unsigned long long int d) { return pressure::psi(static_cast<double>(d)); }
namespace constants {

// Fundamental constants
inline constexpr auto c = velocity::mps(299792458.0); // Speed of light in vacuum
inline constexpr auto c_f = velocity_f::mps(299792458.0f);

inline constexpr auto h = quantity<double, mult_t<dimensions::energy, dimensions::time>>::from_value(6.62607015e-34); // Planck constant
inline constexpr auto h_f = quantity<float, mult_t<dimensions::energy, dimensions::time>>::from_value(6.62607015e-34f);

inline constexpr auto hbar = quantity<double, mult_t<dimensions::energy, dimensions::time>>::from_value(1.054571817e-34); // Reduced Planck constant
inline constexpr auto hbar_f = quantity<float, mult_t<dimensions::energy, dimensions::time>>::from_value(1.054571817e-34f);

inline constexpr auto G = quantity<double, div_t<mult_t<dimensions::force, dimensions::area>, mult_t<dimensions::mass, dimensions::mass>>>::from_value(6.67430e-11); // Gravitational constant
inline constexpr auto G_f = quantity<float, div_t<mult_t<dimensions::force, dimensions::area>, mult_t<dimensions::mass, dimensions::mass>>>::from_value(6.67430e-11f);

inline constexpr auto e = quantity<double, dimensions::charge>::from_value(1.602176634e-19); // Elementary charge
inline constexpr auto e_f = quantity<float, dimensions::charge>::from_value(1.602176634e-19f);

// Electromagnetic constants
inline constexpr auto mu_0 = quantity<double, div_t<dimensions::force, mult_t<dimensions::current, dimensions::current>>>::from_value(1.25663706212e-6); // Vacuum permeability
inline constexpr auto mu_0_f = quantity<float, div_t<dimensions::force, mult_t<dimensions::current, dimensions::current>>>::from_value(1.25663706212e-6f);

inline constexpr auto epsilon_0 = quantity<double, div_t<dimensions::capacitance, dimensions::length>>::from_value(8.8541878128e-12); // Vacuum permittivity
inline constexpr auto epsilon_0_f = quantity<float, div_t<dimensions::capacitance, dimensions::length>>::from_value(8.8541878128e-12f);

// Atomic and nuclear constants
inline constexpr auto m_e = mass::kilograms(9.1093837015e-31); // Electron mass
inline constexpr auto m_e_f = mass_f::kilograms(9.1093837015e-31f);

inline constexpr auto m_p = mass::kilograms(1.67262192369e-27); // Proton mass
inline constexpr auto m_p_f = mass_f::kilograms(1.67262192369e-27f);

inline constexpr auto m_n = mass::kilograms(1.67492749804e-27); // Neutron mass
inline constexpr auto m_n_f = mass_f::kilograms(1.67492749804e-27f);

inline constexpr auto alpha = dimensionless::from_value(7.2973525693e-3); // Fine-structure constant
inline constexpr auto alpha_f = dimensionless_f::from_value(7.2973525693e-3f);

// Physico-chemical constants
inline constexpr auto N_A = quantity<double, dimensions::inv_t<dimensions::amount_of_substance>>::from_value(6.02214076e23); // Avogadro constant
inline constexpr auto N_A_f = quantity<float, dimensions::inv_t<dimensions::amount_of_substance>>::from_value(6.02214076e23f);

inline constexpr auto k_B = quantity<double, div_t<dimensions::energy, dimensions::temperature>>::from_value(1.380649e-23); // Boltzmann constant
inline constexpr auto k_B_f = quantity<float, div_t<dimensions::energy, dimensions::temperature>>::from_value(1.380649e-23f);

inline constexpr auto R = quantity<double, div_t<mult_t<dimensions::energy, dimensions::amount_of_substance>, dimensions::temperature>>::from_value(8.31446261815324); // Gas constant
inline constexpr auto R_f = quantity<float, div_t<mult_t<dimensions::energy, dimensions::amount_of_substance>, dimensions::temperature>>::from_value(8.31446261815324f);

// Earth and astronomical constants
inline constexpr auto g = acceleration::mps2(9.80665); // Standard acceleration due to gravity
inline constexpr auto g_f = acceleration_f::mps2(9.80665f);

inline constexpr auto au = length::meters(149597870700.0); // Astronomical unit
inline constexpr auto au_f = length_f::meters(149597870700.0f);

// Mathematical constants
inline constexpr auto pi = dimensionless::from_value(3.141592653589793238462643383279502884);
inline constexpr auto pi_f = dimensionless_f::from_value(3.141592653589793238462643383279502884f);

inline constexpr auto e_const = dimensionless::from_value(2.718281828459045235360287471352662498);
inline constexpr auto e_const_f = dimensionless_f::from_value(2.718281828459045235360287471352662498f);

} // namespace constants
namespace units {

// Length units
inline constexpr auto meter = quantity<double, dimensions::length>::from_value(1.0);
inline constexpr auto meter_f = quantity<float, dimensions::length>::from_value(1.0f);
inline constexpr auto foot = quantity<double, dimensions::length>::feet(1.0);
inline constexpr auto foot_f = quantity<float, dimensions::length>::feet(1.0f);
inline constexpr auto inch = quantity<double, dimensions::length>::inches(1.0);
inline constexpr auto inch_f = quantity<float, dimensions::length>::inches(1.0f);
inline constexpr auto yard = quantity<double, dimensions::length>::yards(1.0);
inline constexpr auto yard_f = quantity<float, dimensions::length>::yards(1.0f);
inline constexpr auto mile = quantity<double, dimensions::length>::miles(1.0);
inline constexpr auto mile_f = quantity<float, dimensions::length>::miles(1.0f);

// Area units
inline constexpr auto square_meter = quantity<double, dimensions::area>::square_meters(1.0);
inline constexpr auto square_meter_f = quantity<float, dimensions::area>::square_meters(1.0f);
inline constexpr auto square_foot = quantity<double, dimensions::area>::square_feet(1.0);
inline constexpr auto square_foot_f = quantity<float, dimensions::area>::square_feet(1.0f);
inline constexpr auto square_inch = quantity<double, dimensions::area>::square_inches(1.0);
inline constexpr auto square_inch_f = quantity<float, dimensions::area>::square_inches(1.0f);
inline constexpr auto square_yard = quantity<double, dimensions::area>::square_yards(1.0);
inline constexpr auto square_yard_f = quantity<float, dimensions::area>::square_yards(1.0f);
inline constexpr auto square_mile = quantity<double, dimensions::area>::square_miles(1.0);
inline constexpr auto square_mile_f = quantity<float, dimensions::area>::square_miles(1.0f);
inline constexpr auto acre = quantity<double, dimensions::area>::acres(1.0);
inline constexpr auto acre_f = quantity<float, dimensions::area>::acres(1.0f);

// Volume units
inline constexpr auto cubic_meter = quantity<double, dimensions::volume>::cubic_meters(1.0);
inline constexpr auto cubic_meter_f = quantity<float, dimensions::volume>::cubic_meters(1.0f);
inline constexpr auto cubic_foot = quantity<double, dimensions::volume>::cubic_feet(1.0);
inline constexpr auto cubic_foot_f = quantity<float, dimensions::volume>::cubic_feet(1.0f);
inline constexpr auto liter = quantity<double, dimensions::volume>::liters(1.0);
inline constexpr auto liter_f = quantity<float, dimensions::volume>::liters(1.0f);
inline constexpr auto gallon = quantity<double, dimensions::volume>::gallons(1.0);
inline constexpr auto gallon_f = quantity<float, dimensions::volume>::gallons(1.0f);

// Time units
inline constexpr auto second = quantity<double, dimensions::time>::seconds(1.0);
inline constexpr auto second_f = quantity<float, dimensions::time>::seconds(1.0f);

// Mass units
inline constexpr auto kilogram = quantity<double, dimensions::mass>::kilograms(1.0);
inline constexpr auto kilogram_f = quantity<float, dimensions::mass>::kilograms(1.0f);
inline constexpr auto slug = quantity<double, dimensions::mass>::slugs(1.0);
inline constexpr auto slug_f = quantity<float, dimensions::mass>::slugs(1.0f);

// Temperature units
inline constexpr auto kelvin = quantity<double, dimensions::temperature>::kelvin(1.0);
inline constexpr auto kelvin_f = quantity<float, dimensions::temperature>::kelvin(1.0f);
inline constexpr auto celsius = quantity<double, dimensions::temperature>::celcius(1.0);
inline constexpr auto celsius_f = quantity<float, dimensions::temperature>::celcius(1.0f);
inline constexpr auto fahrenheit = quantity<double, dimensions::temperature>::fahrenheit(1.0);
inline constexpr auto fahrenheit_f = quantity<float, dimensions::temperature>::fahrenheit(1.0f);

// Flow units
inline constexpr auto cubic_meter_per_second = quantity<double, dimensions::flow>::cms(1.0);
inline constexpr auto cubic_meter_per_second_f = quantity<float, dimensions::flow>::cms(1.0f);
inline constexpr auto cubic_foot_per_second = quantity<double, dimensions::flow>::cfs(1.0);
inline constexpr auto cubic_foot_per_second_f = quantity<float, dimensions::flow>::cfs(1.0f);
inline constexpr auto gallon_per_minute = quantity<double, dimensions::flow>::gpm(1.0);
inline constexpr auto gallon_per_minute_f = quantity<float, dimensions::flow>::gpm(1.0f);
inline constexpr auto gallon_per_day = quantity<double, dimensions::flow>::gpd(1.0);
inline constexpr auto gallon_per_day_f = quantity<float, dimensions::flow>::gpd(1.0f);
inline constexpr auto million_gallon_per_day = quantity<double, dimensions::flow>::mgd(1.0);
inline constexpr auto million_gallon_per_day_f = quantity<float, dimensions::flow>::mgd(1.0f);

// Velocity units
inline constexpr auto meter_per_second = quantity<double, dimensions::velocity>::mps(1.0);
inline constexpr auto meter_per_second_f = quantity<float, dimensions::velocity>::mps(1.0f);
inline constexpr auto foot_per_second = quantity<double, dimensions::velocity>::fps(1.0);
inline constexpr auto foot_per_second_f = quantity<float, dimensions::velocity>::fps(1.0f);

// Acceleration units
inline constexpr auto meter_per_second_squared = quantity<double, dimensions::acceleration>::mps2(1.0);
inline constexpr auto meter_per_second_squared_f = quantity<float, dimensions::acceleration>::mps2(1.0f);
inline constexpr auto foot_per_second_squared = quantity<double, dimensions::acceleration>::fps2(1.0);
inline constexpr auto foot_per_second_squared_f = quantity<float, dimensions::acceleration>::fps2(1.0f);

// Density units
inline constexpr auto kilogram_per_cubic_meter = quantity<double, dimensions::density>::kilograms_per_cubic_meter(1.0);
inline constexpr auto kilogram_per_cubic_meter_f = quantity<float, dimensions::density>::kilograms_per_cubic_meter(1.0f);
inline constexpr auto slug_per_cubic_foot = quantity<double, dimensions::density>::slugs_per_cubic_foot(1.0);
inline constexpr auto slug_per_cubic_foot_f = quantity<float, dimensions::density>::slugs_per_cubic_foot(1.0f);

// Dynamic viscosity units
inline constexpr auto pascal_second = quantity<double, dimensions::dynamic_viscosity>::pascal_seconds(1.0);
inline constexpr auto pascal_second_f = quantity<float, dimensions::dynamic_viscosity>::pascal_seconds(1.0f);
inline constexpr auto pound_per_square_foot_second = quantity<double, dimensions::dynamic_viscosity>::psf_seconds(1.0);
inline constexpr auto pound_per_square_foot_second_f = quantity<float, dimensions::dynamic_viscosity>::psf_seconds(1.0f);

// Kinematic viscosity units
inline constexpr auto square_meter_per_second = quantity<double, dimensions::kinematic_viscosity>::square_meters_per_second(1.0);
inline constexpr auto square_meter_per_second_f = quantity<float, dimensions::kinematic_viscosity>::square_meters_per_second(1.0f);
inline constexpr auto square_foot_per_second = quantity<double, dimensions::kinematic_viscosity>::square_feet_per_second(1.0);
inline constexpr auto square_foot_per_second_f = quantity<float, dimensions::kinematic_viscosity>::square_feet_per_second(1.0f);

// Force units
inline constexpr auto newton = quantity<double, dimensions::force>::newtons(1.0);
inline constexpr auto newton_f = quantity<float, dimensions::force>::newtons(1.0f);
inline constexpr auto pound_force = quantity<double, dimensions::force>::pounds(1.0);
inline constexpr auto pound_force_f = quantity<float, dimensions::force>::pounds(1.0f);

// Force density units
inline constexpr auto newton_per_cubic_meter = quantity<double, dimensions::force_density>::newtons_per_cubic_meter(1.0);
inline constexpr auto newton_per_cubic_meter_f = quantity<float, dimensions::force_density>::newtons_per_cubic_meter(1.0f);
inline constexpr auto pound_per_cubic_foot = quantity<double, dimensions::force_density>::pcf(1.0);
inline constexpr auto pound_per_cubic_foot_f = quantity<float, dimensions::force_density>::pcf(1.0f);

// Dimensionless units
inline constexpr auto radian = quantity<double, dimensions::dimensionless>::radians(1.0);
inline constexpr auto radian_f = quantity<float, dimensions::dimensionless>::radians(1.0f);
inline constexpr auto degree = quantity<double, dimensions::dimensionless>::degrees(1.0);
inline constexpr auto degree_f = quantity<float, dimensions::dimensionless>::degrees(1.0f);

// Pressure units
inline constexpr auto pascal = quantity<double, dimensions::pressure>::pascals(1.0);
inline constexpr auto pascal_f = quantity<float, dimensions::pressure>::pascals(1.0f);
inline constexpr auto pound_per_square_foot = quantity<double, dimensions::pressure>::psf(1.0);
inline constexpr auto pound_per_square_foot_f = quantity<float, dimensions::pressure>::psf(1.0f);
inline constexpr auto pound_per_square_inch = quantity<double, dimensions::pressure>::psi(1.0);
inline constexpr auto pound_per_square_inch_f = quantity<float, dimensions::pressure>::psi(1.0f);

} // namespace units
} // namespace quantities
} // namespace squint
#endif // SQUINT_QUANTITIES_HPP
