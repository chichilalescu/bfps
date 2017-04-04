#ifndef PARTICLES_OUTPUT_HDF5_HPP
#define PARTICLES_OUTPUT_HDF5_HPP

#include <memory>
#include <vector>
#include <hdf5.h>

#include "abstract_particles_output.hpp"
#include "scope_timer.hpp"

template <class real_number, int size_particle_positions, int size_particle_rhs>
class particles_output_hdf5 : public abstract_particles_output<real_number, size_particle_positions, size_particle_rhs>{
    using Parent = abstract_particles_output<real_number, size_particle_positions, size_particle_rhs>;

    const std::string filename;

    hid_t file_id;
    const int total_nb_particles;

public:
    particles_output_hdf5(MPI_Comm in_mpi_com, const std::string in_filename, const int inTotalNbParticles)
            : abstract_particles_output<real_number, size_particle_positions, size_particle_rhs>(in_mpi_com, inTotalNbParticles),
              filename(in_filename),
              file_id(0), total_nb_particles(inTotalNbParticles){

        TIMEZONE("particles_output_hdf5::H5Pcreate");
        hid_t plist_id_par = H5Pcreate(H5P_FILE_ACCESS);
        assert(plist_id_par >= 0);
        int retTest = H5Pset_fapl_mpio(plist_id_par, Parent::getCom(), MPI_INFO_NULL);
        assert(retTest >= 0);

        // Parallel HDF5 write
        file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC | H5F_ACC_DEBUG/*H5F_ACC_EXCL*/, H5P_DEFAULT/*H5F_ACC_RDWR*/, plist_id_par);
        assert(file_id >= 0);
        H5Pclose(plist_id_par);
    }

    ~particles_output_hdf5(){
        TIMEZONE("particles_output_hdf5::H5Dclose");
        int rethdf = H5Fclose(file_id);
        assert(rethdf >= 0);
    }

    void write(const int idx_time_step, const real_number* particles_positions, const real_number* particles_rhs,
               const int nb_particles, const int particles_idx_offset) final{
        TIMEZONE("particles_output_hdf5::write");

        assert(particles_idx_offset < Parent::getTotalNbParticles());
        assert(particles_idx_offset+nb_particles <= Parent::getTotalNbParticles());

        hid_t dset_id = H5Gcreate(file_id, ("dataset"+std::to_string(idx_time_step)).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset_id >= 0);

        static_assert(std::is_same<real_number, double>::value
                      || std::is_same<real_number, float>::value, "real_number must be double or float");
        const hid_t type_id = (sizeof(real_number) == 8?H5T_NATIVE_DOUBLE:H5T_NATIVE_FLOAT);

        {
            const hsize_t datacount[3] = {1, total_nb_particles, size_particle_positions};
            hid_t dataspace = H5Screate_simple(3, datacount, NULL);
            assert(dataspace >= 0);

            hid_t dataset_id = H5Dcreate( dset_id, "state", type_id, dataspace, H5P_DEFAULT,
                                          H5P_DEFAULT, H5P_DEFAULT);
            assert(dataset_id >= 0);

            const hsize_t count[3] = {1, nb_particles, size_particle_positions};
            const hsize_t offset[3] = {0, particles_idx_offset, 0};
            hid_t memspace = H5Screate_simple(3, count, NULL);
            assert(memspace >= 0);

            hid_t filespace = H5Dget_space(dataset_id);
            int rethdf = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
            assert(rethdf >= 0);

            hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
            assert(plist_id >= 0);
            rethdf = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
            assert(rethdf >= 0);

            herr_t	status = H5Dwrite(dataset_id, type_id, memspace, filespace,
                      plist_id, particles_positions);
            assert(status >= 0);
            rethdf = H5Sclose(memspace);
            assert(rethdf >= 0);
            rethdf = H5Dclose(dataset_id);
            assert(rethdf >= 0);
            rethdf = H5Sclose(filespace);
            assert(rethdf >= 0);
        }
        {
            const hsize_t datacount[3] = {1, total_nb_particles, size_particle_rhs};
            hid_t dataspace = H5Screate_simple(3, datacount, NULL);
            assert(dataspace >= 0);

            hid_t dataset_id = H5Dcreate( dset_id, "rhs", type_id, dataspace, H5P_DEFAULT,
                                          H5P_DEFAULT, H5P_DEFAULT);
            assert(dataset_id >= 0);

            const hsize_t count[3] = {1, nb_particles, size_particle_rhs};
            const hsize_t offset[3] = {0, particles_idx_offset, 0};
            hid_t memspace = H5Screate_simple(3, count, NULL);
            assert(memspace >= 0);

            hid_t filespace = H5Dget_space(dataset_id);
            assert(filespace >= 0);
            int rethdf = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
            assert(rethdf >= 0);

            hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
            assert(plist_id >= 0);
            rethdf = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
            assert(rethdf >= 0);

            herr_t	status = H5Dwrite(dataset_id, type_id, memspace, filespace,
                      plist_id, particles_rhs);
            assert(status >= 0);
            rethdf = H5Sclose(memspace);
            assert(rethdf >= 0);
            rethdf = H5Dclose(dataset_id);
            assert(rethdf >= 0);
            rethdf = H5Sclose(filespace);
            assert(rethdf >= 0);
        }
        int rethdf = H5Gclose(dset_id);
        assert(rethdf >= 0);
    }
};

#endif
