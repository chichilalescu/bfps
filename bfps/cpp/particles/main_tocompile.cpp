
#include <mpi.h>
#include <memory>
#include <cassert>
#include <cmath>
#include <cfenv>

#include "particles_system.hpp"
#include "particles_interp_spline.hpp"
#include "abstract_particles_input.hpp"
#include "particles_input_hdf5.hpp"
#include "particles_utils.hpp"

class random_particles : public abstract_particles_input {
    const int nb_particles;
    const double box_width;
    const double lower_limit;
    const double upper_limit;
    int my_rank;
    int nb_processes;

public:
    random_particles(const int in_nb_particles, const double in_box_width,
                     const double in_lower_limit, const double in_upper_limit,
                     const int in_my_rank, const int in_nb_processes)
        : nb_particles(in_nb_particles), box_width(in_box_width),
          lower_limit(in_lower_limit), upper_limit(in_upper_limit),
          my_rank(in_my_rank), nb_processes(in_nb_processes){
    }

    int getTotalNbParticles() final{
        return nb_processes*nb_particles;
    }

    int getLocalNbParticles() final{
        return nb_particles;
    }

    int getNbRhs() final{
        return 1;
    }

    std::unique_ptr<double[]> getMyParticles() final {
        std::unique_ptr<double[]> particles(new double[nb_particles*3]);

        for(int idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
            particles[idx_part*3+IDX_X] = drand48() * box_width;
            particles[idx_part*3+IDX_Y] = drand48() * box_width;
            particles[idx_part*3+IDX_Z] = (drand48() * (upper_limit-lower_limit))
                                            + lower_limit;
        }

        return std::move(particles);
    }

    std::unique_ptr<int[]> getMyParticlesIndexes() final {
        std::unique_ptr<int[]> indexes(new int[nb_particles]);

        for(int idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
            indexes[idx_part] = idx_part + my_rank*nb_particles;
        }

        return std::move(indexes);
    }

    std::vector<std::unique_ptr<double[]>> getMyRhs() final {
        std::vector<std::unique_ptr<double[]>> rhs(1);
        rhs[0].reset(new double[nb_particles*3]);
        std::fill(&rhs[0][0], &rhs[0][nb_particles*3], 0);

        return std::move(rhs);
    }
};

int myrank;

int main(int argc, char** argv){
    feenableexcept(FE_INVALID | FE_OVERFLOW);

    MPI_Init(&argc, &argv);
    {
        int my_rank;
        int nb_processes;
        AssertMpi(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
        AssertMpi(MPI_Comm_size(MPI_COMM_WORLD, &nb_processes));
        myrank = my_rank; // global bfps variable

        const int InterpNbNeighbors = 5;

        const std::array<size_t,3> field_grid_dim{100, 100, 100};
        assert(my_rank < field_grid_dim[0]);
        assert(my_rank < field_grid_dim[1]);
        assert(my_rank < field_grid_dim[2]);

        const double partitionIntervalSize = double(field_grid_dim[IDX_Z])/double(nb_processes);
        const int myPartitionInterval[2] = { int(partitionIntervalSize*my_rank), (my_rank==nb_processes-1?field_grid_dim[IDX_Z]:int(partitionIntervalSize*(my_rank+1)))};


        const std::array<double,3> spatial_box_width{10., 10., 10.};
        const double spatial_partition_width = spatial_box_width[IDX_Z]/double(field_grid_dim[IDX_Z]);
        const double my_spatial_low_limit = myPartitionInterval[0]*spatial_partition_width;
        const double my_spatial_up_limit = myPartitionInterval[1]*spatial_partition_width;

        if(my_rank == 0){
            std::cout << "spatial_box_width = " << spatial_box_width[IDX_X] << " " << spatial_box_width[IDX_Y] << " " << spatial_box_width[IDX_Z] << std::endl;
            std::cout << "spatial_partition_width = " << spatial_partition_width << std::endl;
            std::cout << "my_spatial_low_limit = " << my_spatial_low_limit << std::endl;
            std::cout << "my_spatial_up_limit = " << my_spatial_up_limit << std::endl;
        }

        std::array<size_t,3> local_field_dims;
        local_field_dims[IDX_X] = field_grid_dim[IDX_X];
        local_field_dims[IDX_Y] = field_grid_dim[IDX_Y];
        local_field_dims[IDX_Z] = myPartitionInterval[1]-myPartitionInterval[0];
        std::array<size_t,3> local_field_offset;
        local_field_offset[IDX_X] = 0;
        local_field_offset[IDX_Y] = 0;
        local_field_offset[IDX_Z] = myPartitionInterval[0];

        std::unique_ptr<double[]> field_data(new double[local_field_dims[IDX_X]*local_field_dims[IDX_Y]*local_field_dims[IDX_Z]*3]);
        particles_utils::memzero(field_data.get(), local_field_dims[IDX_X]*local_field_dims[IDX_Y]*local_field_dims[IDX_Z]*3);


        particles_system<particles_interp_spline<InterpNbNeighbors,0>, InterpNbNeighbors> part_sys(field_grid_dim,
                                                                                                spatial_box_width,
                                                                                                spatial_partition_width,
                                                                                                my_spatial_low_limit,
                                                                                                my_spatial_up_limit,
                                                                                                field_data.get(),
                                                                                                local_field_dims,
                                                                                                local_field_offset,
                                                                                                local_field_dims,
                                                                                                MPI_COMM_WORLD);

        int total_nb_particles;
        {
            //const int nb_part_to_generate = 1000;
            //random_particles generator(nb_part_to_generate, spatial_box_width, my_spatial_low_limit,
            //                           my_spatial_up_limit, my_rank, nb_processes);
            std::vector<double> spatial_interval_per_proc(nb_processes+1);
            for(int idx_proc = 0 ; idx_proc < nb_processes ; ++idx_proc){
                spatial_interval_per_proc[idx_proc] = partitionIntervalSize*spatial_partition_width*idx_proc;
                std::cout << "spatial_interval_per_proc[idx_proc] " << spatial_interval_per_proc[idx_proc] << std::endl;
            }
            spatial_interval_per_proc[nb_processes] = spatial_box_width[IDX_Z];
            assert(my_spatial_low_limit == spatial_interval_per_proc[my_rank] || fabs((spatial_interval_per_proc[my_rank]-my_spatial_low_limit)/my_spatial_low_limit) < 1e-13);
            assert(my_spatial_up_limit == spatial_interval_per_proc[my_rank+1] || fabs((spatial_interval_per_proc[my_rank+1]-my_spatial_up_limit)/my_spatial_up_limit) < 1e-13);

            particles_input_hdf5<3,3> generator(MPI_COMM_WORLD, "/home/bbramas/Projects/bfps_runs/particles2/N0288_ptest_1e5_particles.h5",
                                           "tracers0", spatial_interval_per_proc);

            total_nb_particles = generator.getTotalNbParticles();
            part_sys.init(generator);
            // generator might be in a modified state here.
        }


        particles_output_hdf5<3,3> particles_output_writer(MPI_COMM_WORLD, "/tmp/res.hdf5", total_nb_particles);
        particles_output_mpiio<3,3> particles_output_writer_mpi(MPI_COMM_WORLD, "/tmp/res.mpiio", total_nb_particles);


        part_sys.completeLoop(0.1);
        particles_output_writer.save(part_sys.getParticlesPositions(),
                                     part_sys.getParticlesCurrentRhs(),
                                     part_sys.getParticlesIndexes(),
                                     part_sys.getLocalNbParticles(), 1);
        particles_output_writer_mpi.save(part_sys.getParticlesPositions(),
                                     part_sys.getParticlesCurrentRhs(),
                                     part_sys.getParticlesIndexes(),
                                     part_sys.getLocalNbParticles(), 1);

        part_sys.completeLoop(0.1);
        particles_output_writer.save(part_sys.getParticlesPositions(),
                                     part_sys.getParticlesCurrentRhs(),
                                     part_sys.getParticlesIndexes(),
                                     part_sys.getLocalNbParticles(), 2);
        particles_output_writer_mpi.save(part_sys.getParticlesPositions(),
                                     part_sys.getParticlesCurrentRhs(),
                                     part_sys.getParticlesIndexes(),
                                     part_sys.getLocalNbParticles(), 2);
    }
    MPI_Finalize();

    return 0;
}
