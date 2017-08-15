#ifndef PARTICLES_ADAMS_BASHFORTH_HPP
#define PARTICLES_ADAMS_BASHFORTH_HPP

#include <stdexcept>
#include <omp.h>

#include "scope_timer.hpp"
#include "particles_utils.hpp"

template <class partsize_t, class real_number, int size_particle_positions = 3, int size_particle_rhs = 3>
class particles_adams_bashforth {
    static_assert(size_particle_positions == size_particle_rhs,
                  "Not having the same dimension for positions and rhs looks like a bug,"
                  "otherwise comment this assertion.");
public:
    static const int Max_steps = 6;

    void move_particles(real_number*__restrict__ particles_positions,
                        const partsize_t nb_particles,
                        const std::unique_ptr<real_number[]> particles_rhs[],
                        const int nb_rhs, const real_number dt) const{
        TIMEZONE("particles_adams_bashforth::move_particles");

        if(Max_steps < nb_rhs){
            throw std::runtime_error("Error, in bfps particles_adams_bashforth.\n"
                                     "Step in particles_adams_bashforth is too large,"
                                     "you must add formulation up this number or limit the number of steps.");
        }

        // Not needed: TIMEZONE_OMP_INIT_PREPARALLEL(omp_get_max_threads())
#pragma omp parallel default(shared)
        {
            particles_utils::IntervalSplitter<partsize_t> interval(nb_particles,
                                                            omp_get_num_threads(),
                                                            omp_get_thread_num());

            const partsize_t value_start = interval.getMyOffset()*size_particle_positions;
            const partsize_t value_end = (interval.getMyOffset()+interval.getMySize())*size_particle_positions;

            // TODO full unroll + blocking
            switch (nb_rhs){
            case 1:
            {
                const real_number* __restrict__ rhs_0 = particles_rhs[0].get();
                for(partsize_t idx_value = value_start ; idx_value < value_end ; ++idx_value){
                    // dt × [0]
                    particles_positions[idx_value] += dt * rhs_0[idx_value];
                }
            }
                break;
            case 2:
            {
                const real_number* __restrict__ rhs_0 = particles_rhs[0].get();
                const real_number* __restrict__ rhs_1 = particles_rhs[1].get();
                for(partsize_t idx_value = value_start ; idx_value < value_end ; ++idx_value){
                    // dt × (3[0] - [1])/2
                    particles_positions[idx_value]
                            += dt * (3.*rhs_0[idx_value]
                                     - rhs_1[idx_value])/2.;
                }
            }
                break;
            case 3:
            {
                const real_number* __restrict__ rhs_0 = particles_rhs[0].get();
                const real_number* __restrict__ rhs_1 = particles_rhs[1].get();
                const real_number* __restrict__ rhs_2 = particles_rhs[2].get();
                for(partsize_t idx_value = value_start ; idx_value < value_end ; ++idx_value){
                    // dt × (23[0] - 16[1] + 5[2])/12
                    particles_positions[idx_value]
                            += dt * (23.*rhs_0[idx_value]
                                     - 16.*rhs_1[idx_value]
                                     +  5.*rhs_2[idx_value])/12.;
                }
            }
                break;
            case 4:
            {
                const real_number* __restrict__ rhs_0 = particles_rhs[0].get();
                const real_number* __restrict__ rhs_1 = particles_rhs[1].get();
                const real_number* __restrict__ rhs_2 = particles_rhs[2].get();
                const real_number* __restrict__ rhs_3 = particles_rhs[3].get();
                for(partsize_t idx_value = value_start ; idx_value < value_end ; ++idx_value){
                    // dt × (55[0] - 59[1] + 37[2] - 9[3])/24
                    particles_positions[idx_value]
                            += dt * (55.*rhs_0[idx_value]
                                     - 59.*rhs_1[idx_value]
                                     + 37.*rhs_2[idx_value]
                                     -  9.*rhs_3[idx_value])/24.;
                }
            }
                break;
            case 5:
            {
                const real_number* __restrict__ rhs_0 = particles_rhs[0].get();
                const real_number* __restrict__ rhs_1 = particles_rhs[1].get();
                const real_number* __restrict__ rhs_2 = particles_rhs[2].get();
                const real_number* __restrict__ rhs_3 = particles_rhs[3].get();
                const real_number* __restrict__ rhs_4 = particles_rhs[4].get();
                for(partsize_t idx_value = value_start ; idx_value < value_end ; ++idx_value){
                    // dt × (1901[0] - 2774[1] + 2616[2] - 1274[3] + 251[4])/720
                    particles_positions[idx_value]
                            += dt * (1901.*rhs_0[idx_value]
                                     - 2774.*rhs_1[idx_value]
                                     + 2616.*rhs_2[idx_value]
                                     - 1274.*rhs_3[idx_value]
                                     +  251.*rhs_4[idx_value])/720.;
                }
            }
                break;
            case 6:
            {
                const real_number* __restrict__ rhs_0 = particles_rhs[0].get();
                const real_number* __restrict__ rhs_1 = particles_rhs[1].get();
                const real_number* __restrict__ rhs_2 = particles_rhs[2].get();
                const real_number* __restrict__ rhs_3 = particles_rhs[3].get();
                const real_number* __restrict__ rhs_4 = particles_rhs[4].get();
                const real_number* __restrict__ rhs_5 = particles_rhs[5].get();
                for(partsize_t idx_value = value_start ; idx_value < value_end ; ++idx_value){
                    // dt × (4277[0] - 7923[1] + 9982[2] - 7298[3] + 2877[4] - 475[5])/1440
                    particles_positions[idx_value]
                            += dt * (4277.*rhs_0[idx_value]
                                     - 7923.*rhs_1[idx_value]
                                     + 9982.*rhs_2[idx_value]
                                     - 7298.*rhs_3[idx_value]
                                     + 2877.*rhs_4[idx_value]
                                     -  475.*rhs_5[idx_value])/1440.;
                }
            }
                break;
            }
        }
    }
};



#endif
