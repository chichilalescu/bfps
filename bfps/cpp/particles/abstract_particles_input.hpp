#ifndef ABSTRACT_PARTICLES_INPUT_HPP
#define ABSTRACT_PARTICLES_INPUT_HPP

#include <tuple>


class abstract_particles_input {
public:
    virtual ~abstract_particles_input(){}

    virtual int getTotalNbParticles()  = 0;
    virtual int getLocalNbParticles()  = 0;
    virtual int getNbRhs()  = 0;

    virtual std::unique_ptr<double[]> getMyParticles()  = 0;
    virtual std::unique_ptr<int[]> getMyParticlesIndexes()  = 0;
    virtual std::vector<std::unique_ptr<double[]>> getMyRhs()  = 0;
};


#endif
