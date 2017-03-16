#ifndef PARTICLES_INTER_SPLINE_HPP
#define PARTICLES_INTER_SPLINE_HPP

template <int interp_neighbours, int mode>
class particles_interp_spline;

#include "spline_n1.hpp"

template <>
class particles_interp_spline<1,0>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n1_m0(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_spline<1,1>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n1_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_spline<1,2>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n1_m2(in_derivative, in_part_val, poly_val);
    }
};

#include "spline_n2.hpp"

template <>
class particles_interp_spline<2,0>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n2_m0(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_spline<2,1>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n2_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_spline<2,2>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n2_m2(in_derivative, in_part_val, poly_val);
    }
};

#include "spline_n3.hpp"

template <>
class particles_interp_spline<3,0>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n3_m0(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_spline<3,1>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n3_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_spline<3,2>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n3_m2(in_derivative, in_part_val, poly_val);
    }
};

#include "spline_n4.hpp"

template <>
class particles_interp_spline<4,0>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n4_m0(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_spline<4,1>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n4_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_spline<4,2>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n4_m2(in_derivative, in_part_val, poly_val);
    }
};

#include "spline_n5.hpp"

template <>
class particles_interp_spline<5,0>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n5_m0(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_spline<5,1>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n5_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_spline<5,2>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n5_m2(in_derivative, in_part_val, poly_val);
    }
};

#include "spline_n6.hpp"

template <>
class particles_interp_spline<6,0>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n6_m0(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_spline<6,1>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n6_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_interp_spline<6,2>{
public:
    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n6_m2(in_derivative, in_part_val, poly_val);
    }
};



#endif
