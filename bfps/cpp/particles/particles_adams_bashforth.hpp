#ifndef PARTICLES_ADAMS_BASHFORTH_HPP
#define PARTICLES_ADAMS_BASHFORTH_HPP

#include <stdexcept>
#include "scope_timer.hpp"

template <class real_number, int size_particle_positions = 3, int size_particle_rhs = 3>
class particles_adams_bashforth {
public:
    static const int Max_steps = 6;

    void move_particles(real_number particles_positions[],
                       const int nb_particles,
                       const std::unique_ptr<real_number[]> particles_rhs[],
                       const int nb_rhs, const real_number dt) const{
        TIMEZONE("particles_adams_bashforth::move_particles");
        // TODO full unroll + blocking
        switch (nb_rhs){
        case 1:
            for(int idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
                for(int idx_dim = 0 ; idx_dim < size_particle_positions ; ++idx_dim){
                    // dt × [0]
                    particles_positions[idx_part*size_particle_positions + idx_dim]
                            += dt * particles_rhs[0][idx_part*size_particle_rhs + idx_dim];
                }
            }
            break;
        case 2:
            for(int idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
                for(int idx_dim = 0 ; idx_dim < size_particle_positions ; ++idx_dim){
                    // dt × (3[0] - [1])/2
                    particles_positions[idx_part*size_particle_positions + idx_dim]
                            += dt * (3.*particles_rhs[0][idx_part*size_particle_rhs + idx_dim]
                                      - particles_rhs[1][idx_part*size_particle_rhs + idx_dim])/2.;
                }
            }
            break;
        case 3:
            for(int idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
                for(int idx_dim = 0 ; idx_dim < size_particle_positions ; ++idx_dim){
                    // dt × (23[0] - 16[1] + 5[2])/12
                    particles_positions[idx_part*size_particle_positions + idx_dim]
                            += dt * (23.*particles_rhs[0][idx_part*size_particle_rhs + idx_dim]
                                   - 16.*particles_rhs[1][idx_part*size_particle_rhs + idx_dim]
                                   +  5.*particles_rhs[2][idx_part*size_particle_rhs + idx_dim])/12.;
                }
            }
            break;
        case 4:
            for(int idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
                for(int idx_dim = 0 ; idx_dim < size_particle_positions ; ++idx_dim){
                    // dt × (55[0] - 59[1] + 37[2] - 9[3])/24
                    particles_positions[idx_part*size_particle_positions + idx_dim]
                            += dt * (55.*particles_rhs[0][idx_part*size_particle_rhs + idx_dim]
                                   - 59.*particles_rhs[1][idx_part*size_particle_rhs + idx_dim]
                                   + 37.*particles_rhs[2][idx_part*size_particle_rhs + idx_dim]
                                   -  9.*particles_rhs[3][idx_part*size_particle_rhs + idx_dim])/24.;
                }
            }
            break;
        case 5:
            for(int idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
                for(int idx_dim = 0 ; idx_dim < size_particle_positions ; ++idx_dim){
                    // dt × (1901[0] - 2774[1] + 2616[2] - 1274[3] + 251[4])/720
                    particles_positions[idx_part*size_particle_positions + idx_dim]
                            += dt * (1901.*particles_rhs[0][idx_part*size_particle_rhs + idx_dim]
                                   - 2774.*particles_rhs[1][idx_part*size_particle_rhs + idx_dim]
                                   + 2616.*particles_rhs[2][idx_part*size_particle_rhs + idx_dim]
                                   - 1274.*particles_rhs[3][idx_part*size_particle_rhs + idx_dim]
                                   +  251.*particles_rhs[4][idx_part*size_particle_rhs + idx_dim])/720.;
                }
            }
            break;
        case 6:
            for(int idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
                for(int idx_dim = 0 ; idx_dim < size_particle_positions ; ++idx_dim){
                    // dt × (4277[0] - 7923[1] + 9982[2] - 7298[3] + 2877[4] - 475[5])/1440
                    particles_positions[idx_part*size_particle_positions + idx_dim]
                            += dt * (4277.*particles_rhs[0][idx_part*size_particle_rhs + idx_dim]
                                   - 7923.*particles_rhs[1][idx_part*size_particle_rhs + idx_dim]
                                   + 9982.*particles_rhs[2][idx_part*size_particle_rhs + idx_dim]
                                   - 7298.*particles_rhs[3][idx_part*size_particle_rhs + idx_dim]
                                   + 2877.*particles_rhs[4][idx_part*size_particle_rhs + idx_dim]
                                   -  475.*particles_rhs[5][idx_part*size_particle_rhs + idx_dim])/1440.;
                }
            }
            break;
        default:
            throw std::runtime_error("Error, in bfps particles_adams_bashforth.\n"
                                     "Step in particles_adams_bashforth is too large,"
                                     "you must add formulation up this number or limit the number of steps.");
        }
    }
};



#endif
