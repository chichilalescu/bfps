#ifndef ABSTRACT_PARTICLES_INPUT_HPP
#define ABSTRACT_PARTICLES_INPUT_HPP

#include <tuple>

template <class partsize_t, class real_number>
class abstract_particles_input {
public:
    virtual ~abstract_particles_input(){}

    virtual partsize_t getTotalNbParticles()  = 0;
    virtual partsize_t getLocalNbParticles()  = 0;
    virtual int getNbRhs()  = 0;

    virtual std::unique_ptr<real_number[]> getMyParticles()  = 0;
    virtual std::unique_ptr<partsize_t[]> getMyParticlesIndexes()  = 0;
    virtual std::vector<std::unique_ptr<real_number[]>> getMyRhs()  = 0;
};


#endif
