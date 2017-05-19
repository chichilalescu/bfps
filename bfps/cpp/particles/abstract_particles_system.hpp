#ifndef ABSTRACT_PARTICLES_SYSTEM_HPP
#define ABSTRACT_PARTICLES_SYSTEM_HPP

#include <memory>

//- Not generic to enable sampling begin
#include "field.hpp"
#include "kspace.hpp"
//- Not generic to enable sampling end


template <class partsize_t, class real_number>
class abstract_particles_system {
public:
    virtual void compute() = 0;

    virtual void move(const real_number dt) = 0;

    virtual void redistribute() = 0;

    virtual void inc_step_idx() = 0;

    virtual void shift_rhs_vectors() = 0;

    virtual void completeLoop(const real_number dt) = 0;

    virtual const real_number* getParticlesPositions() const = 0;

    virtual const std::unique_ptr<real_number[]>* getParticlesRhs() const = 0;

    virtual const partsize_t* getParticlesIndexes() const = 0;

    virtual partsize_t getLocalNbParticles() const = 0;

    virtual partsize_t getGlobalNbParticles() const = 0;

    virtual int getNbRhs() const = 0;

    virtual int get_step_idx() const = 0;

    //- Not generic to enable sampling begin
    virtual void sample_compute_field(const field<float, FFTW, ONE>& sample_field,
                                real_number sample_rhs[]) = 0;
    virtual void sample_compute_field(const field<float, FFTW, THREE>& sample_field,
                                real_number sample_rhs[]) = 0;
    virtual void sample_compute_field(const field<float, FFTW, THREExTHREE>& sample_field,
                                real_number sample_rhs[]) = 0;
    virtual void sample_compute_field(const field<double, FFTW, ONE>& sample_field,
                                real_number sample_rhs[]) = 0;
    virtual void sample_compute_field(const field<double, FFTW, THREE>& sample_field,
                                real_number sample_rhs[]) = 0;
    virtual void sample_compute_field(const field<double, FFTW, THREExTHREE>& sample_field,
                                real_number sample_rhs[]) = 0;
    //- Not generic to enable sampling end
};

#endif
