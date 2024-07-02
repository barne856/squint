#include "squint/dimension.hpp"
#include <iostream>
#include <type_traits>

// Helper function to print dimension exponents
template<typename D>
void print_dimension() {
    std::cout << "L: " << D::L::num << "/" << D::L::den
              << ", T: " << D::T::num << "/" << D::T::den
              << ", M: " << D::M::num << "/" << D::M::den
              << ", K: " << D::K::num << "/" << D::K::den
              << ", I: " << D::I::num << "/" << D::I::den
              << ", N: " << D::N::num << "/" << D::N::den
              << ", J: " << D::J::num << "/" << D::J::den << std::endl;
}

int main() {
    namespace dim = squint::dimensions;

    std::cout << "Testing base dimensions:" << std::endl;
    print_dimension<dim::length>();
    print_dimension<dim::time>();
    print_dimension<dim::mass>();
    print_dimension<dim::temperature>();
    print_dimension<dim::current>();
    print_dimension<dim::amount_of_substance>();
    print_dimension<dim::luminous_intensity>();

    std::cout << "\nTesting derived dimensions:" << std::endl;
    print_dimension<dim::velocity>();
    print_dimension<dim::acceleration>();
    print_dimension<dim::force>();

    std::cout << "\nTesting dimension operations:" << std::endl;
    using energy_from_force = squint::mult_t<dim::force, dim::length>;
    std::cout << "Energy (from force * length): ";
    print_dimension<energy_from_force>();
    std::cout << "Energy (predefined): ";
    print_dimension<dim::energy>();
    std::cout << "Are they the same? " << std::boolalpha 
              << std::is_same_v<energy_from_force, dim::energy> << std::endl;

    using acceleration_from_force = squint::div_t<dim::force, dim::mass>;
    std::cout << "\nAcceleration (from force / mass): ";
    print_dimension<acceleration_from_force>();
    std::cout << "Acceleration (predefined): ";
    print_dimension<dim::acceleration>();
    std::cout << "Are they the same? " << std::boolalpha 
              << std::is_same_v<acceleration_from_force, dim::acceleration> << std::endl;

    using area_squared = squint::pow_t<dim::area, 2>;
    std::cout << "\nArea squared: ";
    print_dimension<area_squared>();

    using length_from_area = squint::root_t<dim::area, 2>;
    std::cout << "Length (from sqrt of area): ";
    print_dimension<length_from_area>();
    std::cout << "Is it the same as base length? " << std::boolalpha 
              << std::is_same_v<length_from_area, dim::length> << std::endl;

    std::cout << "\nTesting new dimensions:" << std::endl;
    print_dimension<dim::magnetic_flux>();
    print_dimension<dim::capacitance>();
    print_dimension<dim::inductance>();

    return 0;
}