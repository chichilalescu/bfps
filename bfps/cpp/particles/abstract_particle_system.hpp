#ifndef ABSTRACT_PARTICLE_SYSTEM_HPP
#define ABSTRACT_PARTICLE_SYSTEM_HPP

class abstract_particle_system {
public:
    virtual void compute() = 0;

    virtual void move(const double dt) = 0;

    virtual void redistribute() = 0;

    virtual void inc_step_idx() = 0;

    virtual void shift_rhs_vectors() = 0;

    virtual void completeLoop(const double dt) = 0;

    virtual const double* getParticlesPositions() const = 0;

    virtual const double* getParticlesCurrentRhs() const = 0;

    virtual const int* getParticlesIndexes() const = 0;

    virtual int getLocalNbParticles() const = 0;
};

#endif
