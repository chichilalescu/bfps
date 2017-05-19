#ifndef PARTICLES_SYSTEM_HPP
#define PARTICLES_SYSTEM_HPP

#include <array>

#include "abstract_particles_system.hpp"
#include "particles_distr_mpi.hpp"
#include "particles_output_hdf5.hpp"
#include "particles_output_mpiio.hpp"
#include "particles_field_computer.hpp"
#include "abstract_particles_input.hpp"
#include "particles_adams_bashforth.hpp"
#include "scope_timer.hpp"

template <class partsize_t, class real_number, class field_rnumber, class field_class, class interpolator_class, int interp_neighbours,
          int size_particle_rhs>
class particles_system : public abstract_particles_system<partsize_t, real_number> {
    MPI_Comm mpi_com;

    const std::pair<int,int> current_partition_interval;
    const int partition_interval_size;

    interpolator_class interpolator;

    particles_distr_mpi<partsize_t, real_number> particles_distr;

    particles_adams_bashforth<partsize_t, real_number, 3, size_particle_rhs> positions_updater;

    using computer_class = particles_field_computer<partsize_t, real_number, interpolator_class, interp_neighbours>;
    computer_class computer;

    field_class default_field;

    std::unique_ptr<partsize_t[]> current_my_nb_particles_per_partition;
    std::unique_ptr<partsize_t[]> current_offset_particles_for_partition;

    const std::array<real_number,3> spatial_box_width;
    const std::array<real_number,3> spatial_partition_width;
    const real_number my_spatial_low_limit;
    const real_number my_spatial_up_limit;

    std::unique_ptr<real_number[]> my_particles_positions;
    std::unique_ptr<partsize_t[]> my_particles_positions_indexes;
    partsize_t my_nb_particles;
    std::vector<std::unique_ptr<real_number[]>> my_particles_rhs;

    int step_idx;

public:
    particles_system(const std::array<size_t,3>& field_grid_dim, const std::array<real_number,3>& in_spatial_box_width,
                     const std::array<real_number,3>& in_spatial_box_offset,
                     const std::array<real_number,3>& in_spatial_partition_width,
                     const real_number in_my_spatial_low_limit, const real_number in_my_spatial_up_limit,
                     const std::array<size_t,3>& in_local_field_dims,
                     const std::array<size_t,3>& in_local_field_offset,
                     const field_class& in_field,
                     MPI_Comm in_mpi_com,
                     const int in_current_iteration = 1)
        : mpi_com(in_mpi_com),
          current_partition_interval({in_local_field_offset[IDX_Z], in_local_field_offset[IDX_Z] + in_local_field_dims[IDX_Z]}),
          partition_interval_size(current_partition_interval.second - current_partition_interval.first),
          interpolator(),
          particles_distr(in_mpi_com, current_partition_interval,field_grid_dim),
          positions_updater(),
          computer(field_grid_dim, current_partition_interval,
                   interpolator, in_spatial_box_width, in_spatial_box_offset, in_spatial_partition_width),
          default_field(in_field),
          spatial_box_width(in_spatial_box_width), spatial_partition_width(in_spatial_partition_width),
          my_spatial_low_limit(in_my_spatial_low_limit), my_spatial_up_limit(in_my_spatial_up_limit),
          my_nb_particles(0), step_idx(in_current_iteration){

        current_my_nb_particles_per_partition.reset(new partsize_t[partition_interval_size]);
        current_offset_particles_for_partition.reset(new partsize_t[partition_interval_size+1]);
    }

    ~particles_system(){
    }

    void init(abstract_particles_input<partsize_t, real_number>& particles_input) {
        TIMEZONE("particles_system::init");

        my_particles_positions = particles_input.getMyParticles();
        my_particles_positions_indexes = particles_input.getMyParticlesIndexes();
        my_particles_rhs = particles_input.getMyRhs();
        my_nb_particles = particles_input.getLocalNbParticles();

        for(partsize_t idx_part = 0 ; idx_part < my_nb_particles ; ++idx_part){ // TODO remove me
            const int partition_level = computer.pbc_field_layer(my_particles_positions[idx_part*3+IDX_Z], IDX_Z);
            assert(partition_level >= current_partition_interval.first);
            assert(partition_level < current_partition_interval.second);
        }

        particles_utils::partition_extra_z<partsize_t, 3>(&my_particles_positions[0], my_nb_particles, partition_interval_size,
                                              current_my_nb_particles_per_partition.get(), current_offset_particles_for_partition.get(),
        [&](const real_number& z_pos){
            const int partition_level = computer.pbc_field_layer(z_pos, IDX_Z);
            assert(current_partition_interval.first <= partition_level && partition_level < current_partition_interval.second);
            return partition_level - current_partition_interval.first;
        },
        [&](const partsize_t idx1, const partsize_t idx2){
            std::swap(my_particles_positions_indexes[idx1], my_particles_positions_indexes[idx2]);
            for(int idx_rhs = 0 ; idx_rhs < int(my_particles_rhs.size()) ; ++idx_rhs){
                for(int idx_val = 0 ; idx_val < size_particle_rhs ; ++idx_val){
                    std::swap(my_particles_rhs[idx_rhs][idx1*size_particle_rhs + idx_val],
                              my_particles_rhs[idx_rhs][idx2*size_particle_rhs + idx_val]);
                }
            }
        });

        {// TODO remove
            for(int idxPartition = 0 ; idxPartition < partition_interval_size ; ++idxPartition){
                assert(current_my_nb_particles_per_partition[idxPartition] ==
                       current_offset_particles_for_partition[idxPartition+1] - current_offset_particles_for_partition[idxPartition]);
                for(partsize_t idx = current_offset_particles_for_partition[idxPartition] ; idx < current_offset_particles_for_partition[idxPartition+1] ; ++idx){
                    assert(computer.pbc_field_layer(my_particles_positions[idx*3+IDX_Z], IDX_Z)-current_partition_interval.first == idxPartition);
                }
            }
        }
    }


    void compute() final {
        TIMEZONE("particles_system::compute");
        particles_distr.template compute_distr<computer_class, field_class, 3, size_particle_rhs>(
                               computer, default_field,
                               current_my_nb_particles_per_partition.get(),
                               my_particles_positions.get(),
                               my_particles_rhs.front().get(),
                               interp_neighbours);
    }

    template <class sample_field_class, int sample_size_particle_rhs>
    void sample_compute(const sample_field_class& sample_field,
                        real_number sample_rhs[]) {
        TIMEZONE("particles_system::compute");
        particles_distr.template compute_distr<computer_class, sample_field_class, 3, sample_size_particle_rhs>(
                               computer, sample_field,
                               current_my_nb_particles_per_partition.get(),
                               my_particles_positions.get(),
                               sample_rhs,
                               interp_neighbours);
    }

    //- Not generic to enable sampling begin
    void sample_compute_field(const field<float, FFTW, ONE>& sample_field,
                                real_number sample_rhs[]) final {
        // sample_compute<decltype(sample_field), 1>(sample_field, sample_rhs);
    }
    void sample_compute_field(const field<float, FFTW, THREE>& sample_field,
                                real_number sample_rhs[]) final {
        sample_compute<decltype(sample_field), 3>(sample_field, sample_rhs);
    }
    void sample_compute_field(const field<float, FFTW, THREExTHREE>& sample_field,
                                real_number sample_rhs[]) final {
        sample_compute<decltype(sample_field), 9>(sample_field, sample_rhs);
    }
    void sample_compute_field(const field<double, FFTW, ONE>& sample_field,
                                real_number sample_rhs[]) final {
        sample_compute<decltype(sample_field), 1>(sample_field, sample_rhs);
    }
    void sample_compute_field(const field<double, FFTW, THREE>& sample_field,
                                real_number sample_rhs[]) final {
        sample_compute<decltype(sample_field), 3>(sample_field, sample_rhs);
    }
    void sample_compute_field(const field<double, FFTW, THREExTHREE>& sample_field,
                                real_number sample_rhs[]) final {
        sample_compute<decltype(sample_field), 9>(sample_field, sample_rhs);
    }
    //- Not generic to enable sampling end

    void move(const real_number dt) final {
        TIMEZONE("particles_system::move");
        positions_updater.move_particles(my_particles_positions.get(), my_nb_particles,
                                my_particles_rhs.data(), std::min(step_idx,int(my_particles_rhs.size())),
                                dt);
    }

    void redistribute() final {
        TIMEZONE("particles_system::redistribute");
        particles_distr.template redistribute<computer_class, 3, size_particle_rhs, 1>(
                              computer,
                              current_my_nb_particles_per_partition.get(),
                              &my_nb_particles,
                              &my_particles_positions,
                              my_particles_rhs.data(), int(my_particles_rhs.size()),
                              &my_particles_positions_indexes);
    }

    void inc_step_idx() final {
        step_idx += 1;
    }

    void shift_rhs_vectors() final {
        if(my_particles_rhs.size()){
            std::unique_ptr<real_number[]> next_current(std::move(my_particles_rhs.back()));
            for(int idx_rhs = int(my_particles_rhs.size())-1 ; idx_rhs > 0 ; --idx_rhs){
                my_particles_rhs[idx_rhs] = std::move(my_particles_rhs[idx_rhs-1]);
            }
            my_particles_rhs[0] = std::move(next_current);
            particles_utils::memzero(my_particles_rhs[0], size_particle_rhs*my_nb_particles);
        }
    }

    void completeLoop(const real_number dt) final {
        TIMEZONE("particles_system::completeLoop");
        compute();
        move(dt);
        redistribute();
        inc_step_idx();
        shift_rhs_vectors();
    }

    const real_number* getParticlesPositions() const final {
        return my_particles_positions.get();
    }

    const std::unique_ptr<real_number[]>* getParticlesRhs() const final {
        return my_particles_rhs.data();
    }

    const partsize_t* getParticlesIndexes() const final {
        return my_particles_positions_indexes.get();
    }

    partsize_t getLocalNbParticles() const final {
        return my_nb_particles;
    }

    int getNbRhs() const final {
        return int(my_particles_rhs.size());
    }

    void checkNan() const { // TODO remove
        for(partsize_t idx_part = 0 ; idx_part < my_nb_particles ; ++idx_part){ // TODO remove me
            assert(std::isnan(my_particles_positions[idx_part*3+IDX_X]) == false);
            assert(std::isnan(my_particles_positions[idx_part*3+IDX_Y]) == false);
            assert(std::isnan(my_particles_positions[idx_part*3+IDX_Z]) == false);

            for(int idx_rhs = 0 ; idx_rhs < my_particles_rhs.size() ; ++idx_rhs){
                for(int idx_rhs_val = 0 ; idx_rhs_val < size_particle_rhs ; ++idx_rhs_val){
                    assert(std::isnan(my_particles_rhs[idx_rhs][idx_part*size_particle_rhs+idx_rhs_val]) == false);
                }
            }
        }
    }
};


#endif
