#ifndef PARTICLES_OUTPUT_SAMPLING_HDF5_HPP
#define PARTICLES_OUTPUT_SAMPLING_HDF5_HPP

#include "abstract_particles_output.hpp"

#include <hdf5.h>

template <class partsize_t,
          class real_number,
          int size_particle_positions,
          int size_particle_rhs>
class particles_output_sampling_hdf5 : public abstract_particles_output<partsize_t,
                                                               real_number,
                                                               size_particle_positions,
                                                               size_particle_rhs>{
    using Parent = abstract_particles_output<partsize_t,
                                             real_number,
                                             size_particle_positions,
                                             size_particle_rhs>;

    hid_t parent_group;
    const std::string dataset_name;
    const bool use_collective_io;

public:
    particles_output_sampling_hdf5(MPI_Comm in_mpi_com,
                          const partsize_t inTotalNbParticles,
                          const hid_t in_parent_group,
                          const std::string& in_dataset_name,
                          const bool in_use_collective_io = false)
            : Parent(in_mpi_com, inTotalNbParticles, 1),
              parent_group(in_parent_group),
              dataset_name(in_dataset_name),
              use_collective_io(in_use_collective_io){}

    void write(
            const int idx_time_step,
            const real_number* /*particles_positions*/,
            const std::unique_ptr<real_number[]>* particles_rhs,
            const partsize_t nb_particles,
            const partsize_t particles_idx_offset) final{
        assert(Parent::isInvolved());

        TIMEZONE("particles_output_hdf5::write");

        assert(particles_idx_offset < Parent::getTotalNbParticles() || (particles_idx_offset == Parent::getTotalNbParticles() && nb_particles == 0));
        assert(particles_idx_offset+nb_particles <= Parent::getTotalNbParticles());

        static_assert(std::is_same<real_number, double>::value ||
                      std::is_same<real_number, float>::value,
                      "real_number must be double or float");
        const hid_t type_id = (sizeof(real_number) == 8 ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT);

        hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
        assert(plist_id >= 0);
        {
            int rethdf = H5Pset_dxpl_mpio(plist_id, use_collective_io ? H5FD_MPIO_COLLECTIVE : H5FD_MPIO_INDEPENDENT);
            assert(rethdf >= 0);
        }
        {
            assert(size_particle_rhs >= 0);
            const hsize_t datacount[3] = {hsize_t(Parent::getNbRhs()),
                                          hsize_t(Parent::getTotalNbParticles()),
                                          hsize_t(size_particle_rhs)};
            hid_t dataspace = H5Screate_simple(3, datacount, NULL);
            assert(dataspace >= 0);

            hid_t dataset_id = H5Dcreate( parent_group,
                                          dataset_name.c_str(),
                                          type_id,
                                          dataspace,
                                          H5P_DEFAULT,
                                          H5P_DEFAULT,
                                          H5P_DEFAULT);
            assert(dataset_id >= 0);

            assert(particles_idx_offset >= 0);
            const hsize_t count[3] = {
                1,
                hsize_t(nb_particles),
                hsize_t(size_particle_rhs)};
            const hsize_t offset[3] = {
                0,
                hsize_t(particles_idx_offset),
                0};
            hid_t memspace = H5Screate_simple(3, count, NULL);
            assert(memspace >= 0);

            hid_t filespace = H5Dget_space(dataset_id);
            assert(filespace >= 0);
            int rethdf = H5Sselect_hyperslab(
                    filespace,
                    H5S_SELECT_SET,
                    offset,
                    NULL,
                    count,
                    NULL);
            assert(rethdf >= 0);

            herr_t	status = H5Dwrite(
                    dataset_id,
                    type_id,
                    memspace,
                    filespace,
                    plist_id,
                    particles_rhs[0].get());
            assert(status >= 0);
            rethdf = H5Sclose(filespace);
            assert(rethdf >= 0);
            rethdf = H5Sclose(memspace);
            assert(rethdf >= 0);
            rethdf = H5Dclose(dataset_id);
            assert(rethdf >= 0);
        }

        {
            int rethdf = H5Pclose(plist_id);
            assert(rethdf >= 0);
        }
    }
};

#endif
