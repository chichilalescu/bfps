#ifndef PARTICLES_GENERIC_INTERP_HPP
#define PARTICLES_GENERIC_INTERP_HPP

template <class real_number, int interp_neighbours, int mode>
class particles_generic_interp;

#include "Lagrange_polys.hpp"
#include "spline.hpp"

template <>
class particles_generic_interp<double, 1,0>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 1,1>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n1_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 1,2>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n1_m2(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 2,0>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n2(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 2,1>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n2_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 2,2>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n2_m2(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 3,0>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n3(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 3,1>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n3_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 3,2>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n3_m2(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 4,0>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n4(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 4,1>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n4_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 4,2>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n4_m2(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 5,0>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n5(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 5,1>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n5_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 5,2>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n5_m2(in_derivative, in_part_val, poly_val);
    }
};


template <>
class particles_generic_interp<double, 6,0>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n6(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 6,1>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n6_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 6,2>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n6_m2(in_derivative, in_part_val, poly_val);
    }
};


template <>
class particles_generic_interp<double, 7,0>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n7(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 7,1>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n7_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 7,2>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n7_m2(in_derivative, in_part_val, poly_val);
    }
};


template <>
class particles_generic_interp<double, 8,0>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n8(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 8,1>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n8_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 8,2>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n8_m2(in_derivative, in_part_val, poly_val);
    }
};


template <>
class particles_generic_interp<double, 9, 0>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n9(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 9,1>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n9_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 9,2>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n9_m2(in_derivative, in_part_val, poly_val);
    }
};


template <>
class particles_generic_interp<double, 10,0>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_Lagrange_n10(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 10,1>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n10_m1(in_derivative, in_part_val, poly_val);
    }
};

template <>
class particles_generic_interp<double, 10,2>{
public:
    using real_number = double;

    void compute_beta(const int in_derivative, const double in_part_val, double poly_val[]) const {
        beta_n10_m2(in_derivative, in_part_val, poly_val);
    }
};

#endif//PARTICLES_INTERP_SPLINE_HPP

