#ifndef PARTICLES_INTERP_LAGRANGE_HPP
#define PARTICLES_INTERP_LAGRANGE_HPP

template <class real_number, int interp_neighbours>
class particles_interp_lagrange;

#include "Lagrange_polys.hpp"

template <>
class particles_interp_lagrange<double, 1>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_lagrange<double, 2>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n2(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_lagrange<double, 3>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n3(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_lagrange<double, 4>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n4(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_lagrange<double, 5>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n5(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_lagrange<double, 6>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n6(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_lagrange<double, 7>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n7(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_lagrange<double, 8>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n8(in_derivative, in_part_val, poly_val);
    }
};


#endif//PARTICLES_INTERP_LAGRANGE_HPP
