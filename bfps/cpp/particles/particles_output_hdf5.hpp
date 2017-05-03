#ifndef PARTICLES_OUTPUT_HDF5_HPP
#define PARTICLES_OUTPUT_HDF5_HPP

#include <memory>
#include <vector>
#include <hdf5.h>

#include "abstract_particles_output.hpp"
#include "scope_timer.hpp"

template <class real_number,
          int size_particle_positions,
          int size_particle_rhs>
class particles_output_hdf5 : public abstract_particles_output<real_number,
                                                               size_particle_positions,
                                                               size_particle_rhs>{
    using Parent = abstract_particles_output<real_number,
                                             size_particle_positions,
                                             size_particle_rhs>;

    const std::string particle_species_name;

    hid_t file_id;
    const int total_nb_particles;

    hid_t dset_id_state;
    hid_t dset_id_rhs;

public:
    particles_output_hdf5(MPI_Comm in_mpi_com,
                          const std::string ps_name,
                          const int inTotalNbParticles,
                          const int in_nb_rhs)
            : abstract_particles_output<real_number,
                                        size_particle_positions,
                                        size_particle_rhs>(
                                                in_mpi_com,
                                                inTotalNbParticles,
                                                in_nb_rhs),
              particle_species_name(ps_name),
              file_id(0),
              total_nb_particles(inTotalNbParticles),
              dset_id_state(0),
              dset_id_rhs(0){}

    int open_file(std::string filename){
        if(Parent::isInvolved()){
            TIMEZONE("particles_output_hdf5::open_file");

            this->require_checkpoint_groups(filename);

            hid_t plist_id_par = H5Pcreate(H5P_FILE_ACCESS);
            assert(plist_id_par >= 0);
            int retTest = H5Pset_fapl_mpio(
                    plist_id_par,
                    Parent::getComWriter(),
                    MPI_INFO_NULL);
            assert(retTest >= 0);

            // Parallel HDF5 write
            file_id = H5Fopen(
                    filename.c_str(),
                    H5F_ACC_RDWR | H5F_ACC_DEBUG,
                    plist_id_par);
            // file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC | H5F_ACC_DEBUG/*H5F_ACC_EXCL*/, H5P_DEFAULT/*H5F_ACC_RDWR*/, plist_id_par);
            assert(file_id >= 0);
            H5Pclose(plist_id_par);

            dset_id_state = H5Gopen(
                    file_id,
                    (this->particle_species_name + std::string("/state")).c_str(),
                    H5P_DEFAULT);
            assert(dset_id_state >= 0);
            dset_id_rhs = H5Gopen(
                    file_id,
                    (this->particle_species_name + std::string("/rhs")).c_str(),
                    H5P_DEFAULT);
            assert(dset_id_rhs >= 0);
        }
        return EXIT_SUCCESS;
    }

    ~particles_output_hdf5(){}

    int close_file(void){
        if(Parent::isInvolved()){
            TIMEZONE("particles_output_hdf5::close_file");

            int rethdf = H5Gclose(dset_id_state);
            assert(rethdf >= 0);

            rethdf = H5Gclose(dset_id_rhs);
            assert(rethdf >= 0);

            rethdf = H5Fclose(file_id);
            assert(rethdf >= 0);
        }
        return EXIT_SUCCESS;
    }

    void require_checkpoint_groups(std::string filename){
        if(Parent::isInvolved()){
            if (Parent::getMyRank() == 0)
            {
                hid_t file_id = H5Fopen(
                        filename.c_str(),
                        H5F_ACC_RDWR | H5F_ACC_DEBUG,
                        H5P_DEFAULT);
                assert(file_id >= 0);
                bool group_exists = H5Lexists(
                        file_id,
                        this->particle_species_name.c_str(),
                        H5P_DEFAULT);
                if (!group_exists)
                {
                    hid_t gg = H5Gcreate(
                        file_id,
                        this->particle_species_name.c_str(),
                        H5P_DEFAULT,
                        H5P_DEFAULT,
                        H5P_DEFAULT);
                    assert(gg >= 0);
                    H5Gclose(gg);
                }
                hid_t gg = H5Gopen(
                        file_id,
                        this->particle_species_name.c_str(),
                        H5P_DEFAULT);
                assert(gg >= 0);
                group_exists = H5Lexists(
                        gg,
                        "state",
                        H5P_DEFAULT);
                if (!group_exists)
                {
                    hid_t ggg = H5Gcreate(
                        gg,
                        "state",
                        H5P_DEFAULT,
                        H5P_DEFAULT,
                        H5P_DEFAULT);
                    assert(ggg >= 0);
                    H5Gclose(ggg);
                }
                group_exists = H5Lexists(
                        gg,
                        "rhs",
                        H5P_DEFAULT);
                if (!group_exists)
                {
                    hid_t ggg = H5Gcreate(
                        gg,
                        "rhs",
                        H5P_DEFAULT,
                        H5P_DEFAULT,
                        H5P_DEFAULT);
                    assert(ggg >= 0);
                    H5Gclose(ggg);
                }
                H5Gclose(gg);
                H5Fclose(file_id);
            }
            MPI_Barrier(Parent::getComWriter());
        }
    }

    void write(
            const int idx_time_step,
            const real_number* particles_positions,
            const std::unique_ptr<real_number[]>* particles_rhs,
            const int nb_particles,
            const int particles_idx_offset) final{
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
            int rethdf = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
            assert(rethdf >= 0);
        }

        {
            assert(total_nb_particles >= 0);
            assert(size_particle_positions >= 0);
            const hsize_t datacount[2] = {
                hsize_t(total_nb_particles),
                hsize_t(size_particle_positions)};
            hid_t dataspace = H5Screate_simple(2, datacount, NULL);
            assert(dataspace >= 0);

            hid_t dataset_id = H5Dcreate( dset_id_state,
                                          std::to_string(idx_time_step).c_str(),
                                          type_id,
                                          dataspace,
                                          H5P_DEFAULT,
                                          H5P_DEFAULT,
                                          H5P_DEFAULT);
            assert(dataset_id >= 0);

            assert(nb_particles >= 0);
            assert(particles_idx_offset >= 0);
            const hsize_t count[2] = {hsize_t(nb_particles), size_particle_positions};
            const hsize_t offset[2] = {hsize_t(particles_idx_offset), 0};
            hid_t memspace = H5Screate_simple(2, count, NULL);
            assert(memspace >= 0);

            hid_t filespace = H5Dget_space(dataset_id);
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
                    particles_positions);
            assert(status >= 0);
            rethdf = H5Sclose(memspace);
            assert(rethdf >= 0);
            rethdf = H5Dclose(dataset_id);
            assert(rethdf >= 0);
            rethdf = H5Sclose(filespace);
            assert(rethdf >= 0);
        }
        {
            assert(size_particle_rhs >= 0);
            const hsize_t datacount[3] = {hsize_t(Parent::getNbRhs()),
                                          hsize_t(total_nb_particles),
                                          hsize_t(size_particle_rhs)};
            hid_t dataspace = H5Screate_simple(3, datacount, NULL);
            assert(dataspace >= 0);

            hid_t dataset_id = H5Dcreate( dset_id_rhs,
                                          std::to_string(idx_time_step).c_str(),
                                          type_id,
                                          dataspace,
                                          H5P_DEFAULT,
                                          H5P_DEFAULT,
                                          H5P_DEFAULT);
            assert(dataset_id >= 0);

            assert(particles_idx_offset >= 0);
            for(int idx_rhs = 0 ; idx_rhs < Parent::getNbRhs() ; ++idx_rhs){
                const hsize_t count[3] = {
                    1,
                    hsize_t(nb_particles),
                    hsize_t(size_particle_rhs)};
                const hsize_t offset[3] = {
                    hsize_t(idx_rhs),
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
                        particles_rhs[idx_rhs].get());
                assert(status >= 0);
                rethdf = H5Sclose(filespace);
                assert(rethdf >= 0);
                rethdf = H5Sclose(memspace);
                assert(rethdf >= 0);
            }
            int rethdf = H5Dclose(dataset_id);
            assert(rethdf >= 0);
        }

        {
            int rethdf = H5Pclose(plist_id);
            assert(rethdf >= 0);
        }
    }
};

#endif//PARTICLES_OUTPUT_HDF5_HPP

