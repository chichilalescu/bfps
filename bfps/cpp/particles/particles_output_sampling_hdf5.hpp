#ifndef PARTICLES_OUTPUT_SAMPLING_HDF5_HPP
#define PARTICLES_OUTPUT_SAMPLING_HDF5_HPP

#include "abstract_particles_output.hpp"

#include <hdf5.h>

template <class partsize_t,
          class real_number,
          int size_particle_positions,
          int size_particle_rhs>
class particles_output_sampling_hdf5 : public abstract_particles_output<
                                       partsize_t,
                                       real_number,
                                       size_particle_positions,
                                       size_particle_rhs>{
    using Parent = abstract_particles_output<partsize_t,
                                             real_number,
                                             size_particle_positions,
                                             size_particle_rhs>;

    hid_t file_id, pgroup_id;

    std::string dataset_name;
    const bool use_collective_io;

public:
    static bool DatasetExistsCol(MPI_Comm in_mpi_com,
                                  const std::string& in_filename,
                                  const std::string& in_groupname,
                                 const std::string& in_dataset_name){
        int my_rank;
        AssertMpi(MPI_Comm_rank(in_mpi_com, &my_rank));

        int dataset_exists = -1;

        if(my_rank == 0){
            hid_t file_id = H5Fopen(
                    in_filename.c_str(),
                    H5F_ACC_RDWR | H5F_ACC_DEBUG,
                    H5P_DEFAULT);
            assert(file_id >= 0);

            dataset_exists = H5Lexists(
                    file_id,
                    (in_groupname + "/" + in_dataset_name).c_str(),
                    H5P_DEFAULT);

            int retTest = H5Fclose(file_id);
            assert(retTest >= 0);
        }

        AssertMpi(MPI_Bcast( &dataset_exists, 1, MPI_INT, 0, in_mpi_com ));
        return dataset_exists;
    }

    particles_output_sampling_hdf5(
            MPI_Comm in_mpi_com,
            const partsize_t inTotalNbParticles,
            const std::string& in_filename,
            const std::string& in_groupname,
            const std::string& in_dataset_name,
            const bool in_use_collective_io = false)
            : Parent(in_mpi_com, inTotalNbParticles, 1),
              dataset_name(in_dataset_name),
              use_collective_io(in_use_collective_io){
        if(Parent::isInvolved()){
            // prepare parallel MPI access property list
            hid_t plist_id_par = H5Pcreate(H5P_FILE_ACCESS);
            assert(plist_id_par >= 0);
            int retTest = H5Pset_fapl_mpio(
                    plist_id_par,
                    Parent::getComWriter(),
                    MPI_INFO_NULL);
            assert(retTest >= 0);

            // open file for parallel HDF5 access
            file_id = H5Fopen(
                    in_filename.c_str(),
                    H5F_ACC_RDWR | H5F_ACC_DEBUG,
                    plist_id_par);
            assert(file_id >= 0);
            retTest = H5Pclose(plist_id_par);
            assert(retTest >= 0);

            // open group
            pgroup_id = H5Gopen(
                    file_id,
                    in_groupname.c_str(),
                    H5P_DEFAULT);
            assert(pgroup_id >= 0);
        }
    }

    ~particles_output_sampling_hdf5(){
        if(Parent::isInvolved()){
            // close group
            int retTest = H5Gclose(pgroup_id);
            assert(retTest >= 0);
            // close file
            retTest = H5Fclose(file_id);
            assert(retTest >= 0);
        }
    }

    int switch_to_group(
            const std::string &in_groupname)
    {
        if(Parent::isInvolved()){
            // close old group
            int retTest = H5Gclose(pgroup_id);
            assert(retTest >= 0);

            // open new group
            pgroup_id = H5Gopen(
                    file_id,
                    in_groupname.c_str(),
                    H5P_DEFAULT);
            assert(pgroup_id >= 0);
        }
        return EXIT_SUCCESS;
    }

    int save_dataset(
            const std::string& in_groupname,
            const std::string& in_dataset_name,
            const real_number input_particles_positions[],
            const std::unique_ptr<real_number[]> input_particles_rhs[],
            const partsize_t index_particles[],
            const partsize_t nb_particles,
            const int idx_time_step)
    {
        // update group
        int retTest = this->switch_to_group(
                in_groupname);
        assert(retTest == EXIT_SUCCESS);
        // update dataset name
        dataset_name = in_dataset_name + "/" + std::to_string(idx_time_step);
        int dataset_exists;
        if (this->getMyRank() == 0)
            dataset_exists = H5Lexists(
                pgroup_id,
                dataset_name.c_str(),
                H5P_DEFAULT);
        AssertMpi(MPI_Bcast(&dataset_exists, 1, MPI_INT, 0, this->getCom()));
        if (dataset_exists == 0)
            this->save(
                input_particles_positions,
                input_particles_rhs,
                index_particles,
                nb_particles,
                idx_time_step);
        return EXIT_SUCCESS;
    }

    void write(
            const int /*idx_time_step*/,
            const real_number* /*particles_positions*/,
            const std::unique_ptr<real_number[]>* particles_rhs,
            const partsize_t nb_particles,
            const partsize_t particles_idx_offset) final{
        assert(Parent::isInvolved());

        TIMEZONE("particles_output_hdf5::write");

        assert(particles_idx_offset < Parent::getTotalNbParticles() ||
               (particles_idx_offset == Parent::getTotalNbParticles() &&
                nb_particles == 0));
        assert(particles_idx_offset+nb_particles <= Parent::getTotalNbParticles());

        static_assert(std::is_same<real_number, double>::value ||
                      std::is_same<real_number, float>::value,
                      "real_number must be double or float");
        const hid_t type_id = (sizeof(real_number) == 8 ?
                               H5T_NATIVE_DOUBLE :
                               H5T_NATIVE_FLOAT);

        hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
        assert(plist_id >= 0);
        {
            int rethdf = H5Pset_dxpl_mpio(
                    plist_id,
                    (use_collective_io ?
                     H5FD_MPIO_COLLECTIVE :
                     H5FD_MPIO_INDEPENDENT));
            assert(rethdf >= 0);
        }
        {
            assert(size_particle_rhs >= 0);
            const hsize_t datacount[3] = {hsize_t(Parent::getNbRhs()),
                                          hsize_t(Parent::getTotalNbParticles()),
                                          hsize_t(size_particle_rhs)};
            hid_t dataspace = H5Screate_simple(3, datacount, NULL);
            assert(dataspace >= 0);

            hid_t dataset_id = H5Dcreate( pgroup_id,
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
