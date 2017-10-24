#ifndef PARTICLES_INPUT_HDF5_HPP
#define PARTICLES_INPUT_HDF5_HPP

#include <tuple>
#include <mpi.h>
#include <hdf5.h>
#include <cassert>
#include <vector>

#include "abstract_particles_input.hpp"
#include "base.hpp"
#include "alltoall_exchanger.hpp"
#include "particles_utils.hpp"
#include "scope_timer.hpp"


// why is "size_particle_rhs" a template parameter?
// I think it's safe to assume this will always be 3.
template <class partsize_t, class real_number, int size_particle_positions, int size_particle_rhs>
class particles_input_hdf5 : public abstract_particles_input<partsize_t, real_number> {
    const std::string filename;

    MPI_Comm mpi_comm;
    int my_rank;
    int nb_processes;

    hsize_t nb_total_particles;
    hsize_t nb_rhs;
    partsize_t nb_particles_for_me;

    std::unique_ptr<real_number[]> my_particles_positions;
    std::unique_ptr<partsize_t[]> my_particles_indexes;
    std::vector<std::unique_ptr<real_number[]>> my_particles_rhs;

    static std::vector<real_number> BuildLimitsAllProcesses(MPI_Comm mpi_comm,
                                                       const real_number my_spatial_low_limit, const real_number my_spatial_up_limit){
        int my_rank;
        int nb_processes;

        AssertMpi(MPI_Comm_rank(mpi_comm, &my_rank));
        AssertMpi(MPI_Comm_size(mpi_comm, &nb_processes));

        std::vector<real_number> spatial_limit_per_proc(nb_processes*2);

        real_number intervalToSend[2] = {my_spatial_low_limit, my_spatial_up_limit};
        AssertMpi(MPI_Allgather(intervalToSend, 2, particles_utils::GetMpiType(real_number()),
                                spatial_limit_per_proc.data(), 2, particles_utils::GetMpiType(real_number()), mpi_comm));

        for(int idx_proc = 0; idx_proc < nb_processes-1 ; ++idx_proc){
            assert(spatial_limit_per_proc[idx_proc*2] <= spatial_limit_per_proc[idx_proc*2+1]);
            assert(spatial_limit_per_proc[idx_proc*2+1] == spatial_limit_per_proc[(idx_proc+1)*2]);
            spatial_limit_per_proc[idx_proc+1] = spatial_limit_per_proc[idx_proc*2+1];
        }
        spatial_limit_per_proc[nb_processes] = spatial_limit_per_proc[(nb_processes-1)*2+1];
        spatial_limit_per_proc.resize(nb_processes+1);

        return spatial_limit_per_proc;
    }

public:
    particles_input_hdf5(const MPI_Comm in_mpi_comm,const std::string& inFilename,
                         const std::string& inDatanameState, const std::string& inDatanameRhs,
                         const real_number my_spatial_low_limit, const real_number my_spatial_up_limit)
        : particles_input_hdf5(in_mpi_comm, inFilename, inDatanameState, inDatanameRhs,
                               BuildLimitsAllProcesses(in_mpi_comm, my_spatial_low_limit, my_spatial_up_limit)){
    }

    particles_input_hdf5(const MPI_Comm in_mpi_comm,const std::string& inFilename,
                         const std::string& inDatanameState, const std::string& inDatanameRhs,
                         const std::vector<real_number>& in_spatial_limit_per_proc)
        : filename(inFilename),
          mpi_comm(in_mpi_comm), my_rank(-1), nb_processes(-1), nb_total_particles(0),
          nb_particles_for_me(0){
        TIMEZONE("particles_input_hdf5");

        AssertMpi(MPI_Comm_rank(mpi_comm, &my_rank));
        AssertMpi(MPI_Comm_size(mpi_comm, &nb_processes));
        assert(int(in_spatial_limit_per_proc.size()) == nb_processes+1);

        hid_t plist_id_par = H5Pcreate(H5P_FILE_ACCESS);
        assert(plist_id_par >= 0);
        {
            int retTest = H5Pset_fapl_mpio(plist_id_par, mpi_comm, MPI_INFO_NULL);
            assert(retTest >= 0);
        }

        hid_t particle_file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist_id_par);
        assert(particle_file >= 0);

        {
            TIMEZONE("state");
            hid_t dset = H5Dopen(particle_file, inDatanameState.c_str(), H5P_DEFAULT);
            assert(dset >= 0);

            hid_t dspace = H5Dget_space(dset); // copy?
            assert(dspace >= 0);

            hid_t space_dim = H5Sget_simple_extent_ndims(dspace);
            assert(space_dim >= 2);

            std::vector<hsize_t> state_dim_array(space_dim);
            int hdfret = H5Sget_simple_extent_dims(dspace, &state_dim_array[0], NULL);
            assert(hdfret >= 0);
            // Last value is the position dim of the particles
            assert(state_dim_array.back() == size_particle_positions);

            nb_total_particles = 1;
            for (size_t idx_dim = 0; idx_dim < state_dim_array.size()-1; ++idx_dim){
                nb_total_particles *= state_dim_array[idx_dim];
            }

            hdfret = H5Sclose(dspace);
            assert(hdfret >= 0);
            hdfret = H5Dclose(dset);
            assert(hdfret >= 0);
        }
        {
            TIMEZONE("rhs");
            hid_t dset = H5Dopen(particle_file, inDatanameRhs.c_str(), H5P_DEFAULT);
            assert(dset >= 0);
            hid_t dspace = H5Dget_space(dset); // copy?
            assert(dspace >= 0);

            hid_t rhs_dim = H5Sget_simple_extent_ndims(dspace);
            // Chichi comment: this assertion will fail in general, there's no reason for it.
                //assert(rhs_dim == 4);
            std::vector<hsize_t> rhs_dim_array(rhs_dim);

            // Chichi comment: wouldn't &rhs_dim_array.front() be safer?
            int hdfret = H5Sget_simple_extent_dims(dspace, &rhs_dim_array[0], NULL);
            assert(hdfret >= 0);
            assert(rhs_dim_array.back() == size_particle_rhs);
            // Chichi comment: this assertion will fail in general
            //assert(rhs_dim_array.front() == 1);
            nb_rhs = rhs_dim_array[0];

            hdfret = H5Sclose(dspace);
            assert(hdfret >= 0);
            hdfret = H5Dclose(dset);
            assert(hdfret >= 0);
        }

        particles_utils::IntervalSplitter<hsize_t> load_splitter(nb_total_particles, nb_processes, my_rank);

        static_assert(std::is_same<real_number, double>::value
                      || std::is_same<real_number, float>::value, "real_number must be double or float");
        const hid_t type_id = (sizeof(real_number) == 8?H5T_NATIVE_DOUBLE:H5T_NATIVE_FLOAT);

        /// Load the data
        std::unique_ptr<real_number[]> split_particles_positions(new real_number[load_splitter.getMySize()*size_particle_positions]);
        {
            TIMEZONE("state-read");
            hid_t dset = H5Dopen(particle_file, inDatanameState.c_str(), H5P_DEFAULT);
            assert(dset >= 0);

            hid_t rspace = H5Dget_space(dset);
            assert(rspace >= 0);

            hsize_t offset[2] = {load_splitter.getMyOffset(), 0};
            hsize_t mem_dims[2] = {load_splitter.getMySize(), 3};

            hid_t mspace = H5Screate_simple(2, &mem_dims[0], NULL);
            assert(mspace >= 0);

            int rethdf = H5Sselect_hyperslab(rspace, H5S_SELECT_SET, offset,
                                             NULL, mem_dims, NULL);
            assert(rethdf >= 0);
            rethdf = H5Dread(dset, type_id, mspace, rspace, H5P_DEFAULT, split_particles_positions.get());
            assert(rethdf >= 0);

            rethdf = H5Sclose(rspace);
            assert(rethdf >= 0);
            rethdf = H5Dclose(dset);
            assert(rethdf >= 0);
        }
        std::vector<std::unique_ptr<real_number[]>> split_particles_rhs(nb_rhs);
        {
            TIMEZONE("rhs-read");
            hid_t dset = H5Dopen(particle_file, inDatanameRhs.c_str(), H5P_DEFAULT);
            assert(dset >= 0);

            for(hsize_t idx_rhs = 0 ; idx_rhs < nb_rhs ; ++idx_rhs){
                hid_t rspace = H5Dget_space(dset);
                assert(rspace >= 0);

                split_particles_rhs[idx_rhs].reset(new real_number[load_splitter.getMySize()*size_particle_rhs]);

                hsize_t offset[3] = {idx_rhs, load_splitter.getMyOffset(), 0};
                hsize_t mem_dims[3] = {1, load_splitter.getMySize(), size_particle_rhs};

                hid_t mspace = H5Screate_simple( 3, &mem_dims[0], NULL);
                assert(mspace >= 0);

                int rethdf = H5Sselect_hyperslab( rspace, H5S_SELECT_SET, offset,
                                                 NULL, mem_dims, NULL);
                assert(rethdf >= 0);
                rethdf = H5Dread(dset, type_id, mspace, rspace, H5P_DEFAULT, split_particles_rhs[idx_rhs].get());
                assert(rethdf >= 0);

                rethdf = H5Sclose(mspace);
                assert(rethdf >= 0);

                rethdf = H5Sclose(rspace);
                assert(rethdf >= 0);
            }
            int rethdf = H5Dclose(dset);
            assert(rethdf >= 0);
        }

        std::unique_ptr<partsize_t[]> split_particles_indexes(new partsize_t[load_splitter.getMySize()]);
        for(partsize_t idx_part = 0 ; idx_part < partsize_t(load_splitter.getMySize()) ; ++idx_part){
            split_particles_indexes[idx_part] = idx_part + partsize_t(load_splitter.getMyOffset());
        }

        // Permute
        std::vector<partsize_t> nb_particles_per_proc(nb_processes);
        {
            TIMEZONE("partition");

            const real_number spatial_box_offset = in_spatial_limit_per_proc[0];
            const real_number spatial_box_width = in_spatial_limit_per_proc[nb_processes] - in_spatial_limit_per_proc[0];

            partsize_t previousOffset = 0;
            for(int idx_proc = 0 ; idx_proc < nb_processes-1 ; ++idx_proc){
                const real_number limitPartitionShifted = in_spatial_limit_per_proc[idx_proc+1]-spatial_box_offset;
                const partsize_t localOffset = particles_utils::partition_extra<partsize_t, size_particle_positions>(
                                                &split_particles_positions[previousOffset*size_particle_positions],
                                                 partsize_t(load_splitter.getMySize())-previousOffset,
                                                 [&](const real_number val[]){
                    const real_number shiftPos = val[IDX_Z]-spatial_box_offset;
                    const real_number nbRepeat = floor(shiftPos/spatial_box_width);
                    const real_number posInBox = shiftPos - (spatial_box_width*nbRepeat);
                    return posInBox < limitPartitionShifted;
                },
                [&](const partsize_t idx1, const partsize_t idx2){
                    std::swap(split_particles_indexes[idx1], split_particles_indexes[idx2]);
                    for(int idx_rhs = 0 ; idx_rhs < int(nb_rhs) ; ++idx_rhs){
                        for(int idx_val = 0 ; idx_val < size_particle_rhs ; ++idx_val){
                            std::swap(split_particles_rhs[idx_rhs][idx1*size_particle_rhs + idx_val],
                                      split_particles_rhs[idx_rhs][idx2*size_particle_rhs + idx_val]);
                        }
                    }
                }, previousOffset);

                nb_particles_per_proc[idx_proc] = localOffset;
                previousOffset += localOffset;
            }
            nb_particles_per_proc[nb_processes-1] = partsize_t(load_splitter.getMySize()) - previousOffset;
        }

        {
            TIMEZONE("exchanger");
            alltoall_exchanger exchanger(mpi_comm, std::move(nb_particles_per_proc));
            // nb_particles_per_processes cannot be used after due to move
            nb_particles_for_me = exchanger.getTotalToRecv();

            my_particles_positions.reset(new real_number[exchanger.getTotalToRecv()*size_particle_positions]);
            exchanger.alltoallv<real_number>(split_particles_positions.get(), my_particles_positions.get(), size_particle_positions);
            split_particles_positions.release();

            my_particles_indexes.reset(new partsize_t[exchanger.getTotalToRecv()]);
            exchanger.alltoallv<partsize_t>(split_particles_indexes.get(), my_particles_indexes.get());
            split_particles_indexes.release();

            my_particles_rhs.resize(nb_rhs);
            for(int idx_rhs = 0 ; idx_rhs < int(nb_rhs) ; ++idx_rhs){
                my_particles_rhs[idx_rhs].reset(new real_number[exchanger.getTotalToRecv()*size_particle_rhs]);
                exchanger.alltoallv<real_number>(split_particles_rhs[idx_rhs].get(), my_particles_rhs[idx_rhs].get(), size_particle_rhs);
            }
        }

        {
            TIMEZONE("close");
            int hdfret = H5Fclose(particle_file);
            assert(hdfret >= 0);
            hdfret = H5Pclose(plist_id_par);
            assert(hdfret >= 0);
        }
    }

    ~particles_input_hdf5(){
    }

    partsize_t getTotalNbParticles() final{
        return partsize_t(nb_total_particles);
    }

    partsize_t getLocalNbParticles() final{
        return nb_particles_for_me;
    }

    int getNbRhs() final{
        return int(nb_rhs);
    }

    std::unique_ptr<real_number[]> getMyParticles() final {
        assert(my_particles_positions != nullptr);
        return std::move(my_particles_positions);
    }

    std::vector<std::unique_ptr<real_number[]>> getMyRhs() final {
        assert(my_particles_rhs.size() == nb_rhs);
        return std::move(my_particles_rhs);
    }

    std::unique_ptr<partsize_t[]> getMyParticlesIndexes() final {
        assert(my_particles_indexes != nullptr);
        return std::move(my_particles_indexes);
    }
};

#endif
