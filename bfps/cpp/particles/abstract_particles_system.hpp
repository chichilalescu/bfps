#ifndef ABSTRACT_PARTICLES_SYSTEM_HPP
#define ABSTRACT_PARTICLES_SYSTEM_HPP

template <class real_number>
class abstract_particles_system {
public:
    virtual void compute() = 0;

    virtual void move(const real_number dt) = 0;

    virtual void redistribute() = 0;

    virtual void inc_step_idx() = 0;

    virtual void shift_rhs_vectors() = 0;

    virtual void completeLoop(const real_number dt) = 0;

    virtual const real_number* getParticlesPositions() const = 0;

    virtual const real_number* getParticlesCurrentRhs() const = 0;

    virtual const int* getParticlesIndexes() const = 0;

    virtual int getLocalNbParticles() const = 0;
};

#endif
