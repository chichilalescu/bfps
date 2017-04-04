#ifndef PARTICLES_SYSTEM_BUILDER_HPP
#define PARTICLES_SYSTEM_BUILDER_HPP

#include <string>

#include "abstract_particles_system.hpp"
#include "particles_system.hpp"
#include "particles_input_hdf5.hpp"
#include "particles_interp_spline.hpp"

#include "field.hpp"
#include "kspace.hpp"



//////////////////////////////////////////////////////////////////////////////
///
/// Double template "for"
///
//////////////////////////////////////////////////////////////////////////////

namespace Template_double_for_if{

template <class RetType,
          class IterType1, IterType1 CurrentIter1,
          class IterType2, const IterType2 CurrentIter2, const IterType2 iterTo2, const IterType2 IterStep2,
          class Func, bool IsNotOver, typename... Args>
struct For2{
    static RetType evaluate(IterType2 value2, Args... args){
        if(CurrentIter2 == value2){
            return std::move(Func::template instanciate<CurrentIter1, CurrentIter2>(args...));
        }
        else{
            return std::move(For2<RetType,
                                        IterType1, CurrentIter1,
                                        IterType2, CurrentIter2+IterStep2, iterTo2, IterStep2,
                                        Func, (CurrentIter2+IterStep2 < iterTo2), Args...>::evaluate(value2, args...));
        }
    }
};

template <class RetType,
          class IterType1, IterType1 CurrentIter1,
          class IterType2, const IterType2 CurrentIter2, const IterType2 iterTo2, const IterType2 IterStep2,
          class Func, typename... Args>
struct For2<RetType,
                  IterType1, CurrentIter1,
                  IterType2, CurrentIter2, iterTo2, IterStep2,
                  Func, false, Args...>{
    static RetType evaluate(IterType2 value2, Args... args){
        std::cout << __FUNCTION__ << " no matching value found\n";
        return RetType();
    }
};

template <class RetType,
          class IterType1, const IterType1 CurrentIter1, const IterType1 iterTo1, const IterType1 IterStep1,
          class IterType2, const IterType2 IterFrom2, const IterType2 iterTo2, const IterType2 IterStep2,
          class Func, bool IsNotOver, typename... Args>
struct For1{
    static RetType evaluate(IterType1 value1, IterType2 value2, Args... args){
        if(CurrentIter1 == value1){
            return std::move(For2<RetType,
                                        IterType1, CurrentIter1,
                                        IterType2, IterFrom2, iterTo2, IterStep2,
                                        Func, (IterFrom2<iterTo2), Args...>::evaluate(value2, args...));
        }
        else{
            return std::move(For1<RetType,
                              IterType1, CurrentIter1+IterStep1, iterTo1, IterStep1,
                              IterType2, IterFrom2, iterTo2, IterStep2,
                              Func, (CurrentIter1+IterStep1 < iterTo1), Args...>::evaluate(value1, value2, args...));
        }
    }
};

template <class RetType,
          class IterType1, const IterType1 IterFrom1, const IterType1 iterTo1, const IterType1 IterStep1,
          class IterType2, const IterType2 IterFrom2, const IterType2 iterTo2, const IterType2 IterStep2,
          class Func, typename... Args>
struct For1<RetType,
                IterType1, IterFrom1, iterTo1, IterStep1,
                IterType2, IterFrom2, iterTo2, IterStep2,
                Func, false, Args...>{
    static RetType evaluate(IterType1 value1, IterType2 value2, Args... args){
        std::cout << __FUNCTION__ << " no matching value found\n";
        return RetType();
    }
};

template <class RetType,
          class IterType1, const IterType1 IterFrom1, const IterType1 iterTo1, const IterType1 IterStep1,
          class IterType2, const IterType2 IterFrom2, const IterType2 iterTo2, const IterType2 IterStep2,
          class Func, typename... Args>
inline RetType evaluate(IterType1 value1, IterType2 value2, Args... args){
    return std::move(For1<RetType,
            IterType1, IterFrom1, iterTo1, IterStep1,
            IterType2, IterFrom2, iterTo2, IterStep2,
            Func, (IterFrom1<iterTo1), Args...>::evaluate(value1, value2, args...));
}

}


//////////////////////////////////////////////////////////////////////////////
///
/// Builder Functions
///
//////////////////////////////////////////////////////////////////////////////

template <class rnumber, field_backend be>
struct particles_system_build_container {
    template <const int interpolation_size, const int spline_mode>
    static std::unique_ptr<abstract_particles_system<rnumber>> instanciate(
             const field<rnumber, be, THREE>* fs_cvorticity, // (field object)
             const kspace<be, SMOOTH>* fs_kk, // (kspace object, contains dkx, dky, dkz)
             const int nsteps, // to check coherency between parameters and hdf input file (nb rhs)
             const int nparticles, // to check coherency between parameters and hdf input file
             const std::string& fname_input, // particles input filename
             const std::string& dset_name, // dataset name for initial input
             MPI_Comm mpi_comm){

        // The size of the field grid (global size)
        std::array<size_t,3> field_grid_dim;
        field_grid_dim[IDX_X] = fs_cvorticity->rlayout->all_size[0][IDX_X];// nx
        field_grid_dim[IDX_Y] = fs_cvorticity->rlayout->all_size[0][IDX_Y];// nx
        field_grid_dim[IDX_Z] = fs_cvorticity->rlayout->all_size[0][IDX_Z];// nz

        // The size of the local field grid (the field nodes that belong to current process)
        std::array<size_t,3> local_field_dims;
        local_field_dims[IDX_X] = fs_cvorticity->rlayout->subsizes[IDX_X];
        local_field_dims[IDX_Y] = fs_cvorticity->rlayout->subsizes[IDX_Y];
        local_field_dims[IDX_Z] = fs_cvorticity->rlayout->subsizes[IDX_Z];

        // The offset of the local field grid
        std::array<size_t,3> local_field_offset;
        local_field_offset[IDX_X] = fs_cvorticity->rlayout->starts[IDX_X];
        local_field_offset[IDX_Y] = fs_cvorticity->rlayout->starts[IDX_Y];
        local_field_offset[IDX_Z] = fs_cvorticity->rlayout->starts[IDX_Z];
        // Ensure that 1D partitioning is used
        {
            assert(myrank < field_grid_dim[IDX_Z]);
            assert(local_field_offset[IDX_X] == 0);
            assert(local_field_offset[IDX_Y] == 0);
            assert(local_field_dims[IDX_X] == field_grid_dim[IDX_X]);
            assert(local_field_dims[IDX_Y] == field_grid_dim[IDX_Y]);

            int my_rank, nb_processes;
            AssertMpi(MPI_Comm_rank(mpi_comm, &my_rank));
            AssertMpi(MPI_Comm_size(mpi_comm, &nb_processes));
            assert((myrank == 0 && local_field_offset[IDX_Z] == 0)
                   || (myrank != 0 && local_field_offset[IDX_Z] != 0));
            assert((myrank == nprocs-1 && local_field_offset[IDX_Z]+local_field_dims[IDX_Z] == field_grid_dim[IDX_Z])
                   || (myrank != nprocs-1 && local_field_offset[IDX_Z]+local_field_dims[IDX_Z] != field_grid_dim[IDX_Z]));
        }
        // The offset of the local field grid
        std::array<size_t,3> local_field_mem_size;
        local_field_mem_size[IDX_X] = fs_cvorticity->rmemlayout->subsizes[IDX_X];
        local_field_mem_size[IDX_Y] = fs_cvorticity->rmemlayout->subsizes[IDX_Y];
        local_field_mem_size[IDX_Z] = fs_cvorticity->rmemlayout->subsizes[IDX_Z];

        // The spatial box size (all particles should be included inside)
        std::array<rnumber,3> spatial_box_width;
        spatial_box_width[IDX_X] = fs_kk->dkx;
        spatial_box_width[IDX_Y] = fs_kk->dky;
        spatial_box_width[IDX_Z] = fs_kk->dkz;

        // The distance between two field nodes in z
        const rnumber spatial_partition_width_z = spatial_box_width[IDX_Z]/rnumber(field_grid_dim[IDX_Z]);
        // The spatial interval of the current process
        const rnumber my_spatial_low_limit_z = rnumber(local_field_offset[IDX_Z])*spatial_partition_width_z;
        const rnumber my_spatial_up_limit_z = rnumber(local_field_offset[IDX_Z]+local_field_dims[IDX_Z])*spatial_partition_width_z;

        // Create the particles system
        particles_system<rnumber, particles_interp_spline<double, interpolation_size,spline_mode>, interpolation_size>* part_sys
         = new particles_system<rnumber, particles_interp_spline<double, interpolation_size,spline_mode>, interpolation_size>(field_grid_dim,
                                                                                                   spatial_box_width,
                                                                                                   spatial_partition_width_z,
                                                                                                   my_spatial_low_limit_z,
                                                                                                   my_spatial_up_limit_z,
                                                                                                   fs_cvorticity->get_rdata(),
                                                                                                   local_field_dims,
                                                                                                   local_field_offset,
                                                                                                   local_field_mem_size,
                                                                                                   mpi_comm);


        // Load particles from hdf5
        particles_input_hdf5<rnumber, 3,3> generator(mpi_comm, fname_input,
                                            dset_name, my_spatial_low_limit_z, my_spatial_up_limit_z);

        // Ensure parameters match the input file
        if(generator.getNbRhs() != nsteps){
            std::runtime_error(std::string("Nb steps is ") + std::to_string(nsteps)
                               + " in the parameters but " + std::to_string(generator.getNbRhs()) + " in the particles file.");
        }
        // Ensure parameters match the input file
        if(generator.getTotalNbParticles() != nparticles){
            std::runtime_error(std::string("Nb particles is ") + std::to_string(nparticles)
                               + " in the parameters but " + std::to_string(generator.getTotalNbParticles()) + " in the particles file.");
        }

        // Load the particles and move them to the particles system
        part_sys->init(generator);

        // Return the created particles system
        return std::unique_ptr<abstract_particles_system<rnumber>>(part_sys);
    }
};


template <class rnumber, field_backend be>
inline std::unique_ptr<abstract_particles_system<rnumber>> particles_system_builder(
        const field<rnumber, be, THREE>* fs_cvorticity, // (field object)
        const kspace<be, SMOOTH>* fs_kk, // (kspace object, contains dkx, dky, dkz)
        const int nsteps, // to check coherency between parameters and hdf input file (nb rhs)
        const int nparticles, // to check coherency between parameters and hdf input file
        const std::string& fname_input, // particles input filename
        const std::string& dset_name, // dataset name for initial input
        const int interpolation_size,
        const int spline_mode,
        MPI_Comm mpi_comm){
    return Template_double_for_if::evaluate<std::unique_ptr<abstract_particles_system<rnumber>>,
                       int, 1, 7, 1, // interpolation_size
                       int, 0, 3, 1, // spline_mode
                       particles_system_build_container<rnumber,be>>(
                           interpolation_size, // template iterator 1
                           spline_mode, // template iterator 2
                           fs_cvorticity,fs_kk, nsteps, nparticles, fname_input, dset_name, mpi_comm);
}


#endif
