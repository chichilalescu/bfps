#ifndef PARTICLES_SYSTEM_BUILDER_HPP
#define PARTICLES_SYSTEM_BUILDER_HPP

#include <string>

#include "abstract_particles_system.hpp"
#include "particles_system.hpp"
#include "particles_input_hdf5.hpp"
#include "particles_generic_interp.hpp"

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
        std::cout << __FUNCTION__ << "[ERROR] template values for loop 2 " << value2 << " does not exist\n";
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
        std::cout << __FUNCTION__ << "[ERROR] template values for loop 1 " << value1 << " does not exist\n";
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

template <class partsize_t, class field_rnumber, field_backend be, field_components fc, class particles_rnumber>
struct particles_system_build_container {
    template <const int interpolation_size, const int spline_mode>
    static std::unique_ptr<abstract_particles_system<partsize_t, particles_rnumber>> instanciate(
             const field<field_rnumber, be, fc>* fs_field, // (field object)
             const kspace<be, SMOOTH>* fs_kk, // (kspace object, contains dkx, dky, dkz)
             const int nsteps, // to check coherency between parameters and hdf input file (nb rhs)
             const partsize_t nparticles, // to check coherency between parameters and hdf input file
             const std::string& fname_input, // particles input filename
            const std::string& inDatanameState, const std::string& inDatanameRhs, // input dataset names
             MPI_Comm mpi_comm,
            const int in_current_iteration){

        // The size of the field grid (global size) all_size seems
        std::array<size_t,3> field_grid_dim;
        field_grid_dim[IDX_X] = fs_field->rlayout->sizes[FIELD_IDX_X];// nx
        field_grid_dim[IDX_Y] = fs_field->rlayout->sizes[FIELD_IDX_Y];// nx
        field_grid_dim[IDX_Z] = fs_field->rlayout->sizes[FIELD_IDX_Z];// nz

        // The size of the local field grid (the field nodes that belong to current process)
        std::array<size_t,3> local_field_dims;
        local_field_dims[IDX_X] = fs_field->rlayout->subsizes[FIELD_IDX_X];
        local_field_dims[IDX_Y] = fs_field->rlayout->subsizes[FIELD_IDX_Y];
        local_field_dims[IDX_Z] = fs_field->rlayout->subsizes[FIELD_IDX_Z];

        // The offset of the local field grid
        std::array<size_t,3> local_field_offset;
        local_field_offset[IDX_X] = fs_field->rlayout->starts[FIELD_IDX_X];
        local_field_offset[IDX_Y] = fs_field->rlayout->starts[FIELD_IDX_Y];
        local_field_offset[IDX_Z] = fs_field->rlayout->starts[FIELD_IDX_Z];


        // Retreive split from fftw to know processes that have no work
        int my_rank, nb_processes;
        AssertMpi(MPI_Comm_rank(mpi_comm, &my_rank));
        AssertMpi(MPI_Comm_size(mpi_comm, &nb_processes));

        const int split_step = (int(field_grid_dim[IDX_Z])+nb_processes-1)/nb_processes;
        const int nb_processes_involved = (int(field_grid_dim[IDX_Z])+split_step-1)/split_step;

        assert((my_rank < nb_processes_involved && local_field_dims[IDX_Z] != 0)
               || (nb_processes_involved <= my_rank && local_field_dims[IDX_Z] == 0));
        assert(nb_processes_involved <= int(field_grid_dim[IDX_Z]));

        // Make the idle processes starting from the limit (and not 0 as set by fftw)
        if(nb_processes_involved <= my_rank){
            local_field_offset[IDX_Z] = field_grid_dim[IDX_Z];
        }

        // Ensure that 1D partitioning is used
        {
            assert(local_field_offset[IDX_X] == 0);
            assert(local_field_offset[IDX_Y] == 0);
            assert(local_field_dims[IDX_X] == field_grid_dim[IDX_X]);
            assert(local_field_dims[IDX_Y] == field_grid_dim[IDX_Y]);

            assert(my_rank >= nb_processes_involved || ((my_rank == 0 && local_field_offset[IDX_Z] == 0)
                   || (my_rank != 0 && local_field_offset[IDX_Z] != 0)));
            assert(my_rank >= nb_processes_involved || ((my_rank == nb_processes_involved-1 && local_field_offset[IDX_Z]+local_field_dims[IDX_Z] == field_grid_dim[IDX_Z])
                   || (my_rank != nb_processes_involved-1 && local_field_offset[IDX_Z]+local_field_dims[IDX_Z] != field_grid_dim[IDX_Z])));
        }

        // The spatial box size (all particles should be included inside)
        std::array<particles_rnumber,3> spatial_box_width;
        spatial_box_width[IDX_X] = 4 * acos(0) / (fs_kk->dkx);
        spatial_box_width[IDX_Y] = 4 * acos(0) / (fs_kk->dky);
        spatial_box_width[IDX_Z] = 4 * acos(0) / (fs_kk->dkz);

        // Box is in the corner
        std::array<particles_rnumber,3> spatial_box_offset;
        spatial_box_offset[IDX_X] = 0;
        spatial_box_offset[IDX_Y] = 0;
        spatial_box_offset[IDX_Z] = 0;

        // The distance between two field nodes in z
        std::array<particles_rnumber,3> spatial_partition_width;
        spatial_partition_width[IDX_X] = spatial_box_width[IDX_X]/particles_rnumber(field_grid_dim[IDX_X]);
        spatial_partition_width[IDX_Y] = spatial_box_width[IDX_Y]/particles_rnumber(field_grid_dim[IDX_Y]);
        spatial_partition_width[IDX_Z] = spatial_box_width[IDX_Z]/particles_rnumber(field_grid_dim[IDX_Z]);
        // The spatial interval of the current process
        const particles_rnumber my_spatial_low_limit_z = particles_rnumber(local_field_offset[IDX_Z])*spatial_partition_width[IDX_Z];
        const particles_rnumber my_spatial_up_limit_z = particles_rnumber(local_field_offset[IDX_Z]+local_field_dims[IDX_Z])*spatial_partition_width[IDX_Z];

        // Create the particles system
        particles_system<partsize_t, particles_rnumber, field_rnumber, field<field_rnumber, be, fc>, particles_generic_interp<particles_rnumber, interpolation_size,spline_mode>, interpolation_size, ncomp(fc)>>* part_sys
         = new particles_system<partsize_t, particles_rnumber, field_rnumber, field<field_rnumber, be, fc>, particles_generic_interp<particles_rnumber, interpolation_size,spline_mode, interpolation_size, ncomp(fc)>>(field_grid_dim,
                                                                                                   spatial_box_width,
                                                                                                   spatial_box_offset,
                                                                                                   spatial_partition_width,
                                                                                                   my_spatial_low_limit_z,
                                                                                                   my_spatial_up_limit_z,
                                                                                                   local_field_dims,
                                                                                                   local_field_offset,
                                                                                                   (*fs_field),
                                                                                                   mpi_comm,
                                                                                                   in_current_iteration);

        // Load particles from hdf5
        particles_input_hdf5<partsize_t, particles_rnumber, 3,3> generator(mpi_comm, fname_input,
                                            inDatanameState, inDatanameRhs, my_spatial_low_limit_z, my_spatial_up_limit_z);

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

        assert(part_sys->getNbRhs() == nsteps);

        // Return the created particles system
        return std::unique_ptr<abstract_particles_system<partsize_t, particles_rnumber>>(part_sys);
    }
};


template <class partsize_t, class field_rnumber, field_backend be, field_components fc, class particles_rnumber = double>
inline std::unique_ptr<abstract_particles_system<partsize_t, particles_rnumber>> particles_system_builder(
        const field<field_rnumber, be, fc>* fs_field, // (field object)
        const kspace<be, SMOOTH>* fs_kk, // (kspace object, contains dkx, dky, dkz)
        const int nsteps, // to check coherency between parameters and hdf input file (nb rhs)
        const partsize_t nparticles, // to check coherency between parameters and hdf input file
        const std::string& fname_input, // particles input filename
        const std::string& inDatanameState, const std::string& inDatanameRhs, // input dataset names
        const int interpolation_size,
        const int spline_mode,
        MPI_Comm mpi_comm,
        const int in_current_iteration){
    return Template_double_for_if::evaluate<std::unique_ptr<abstract_particles_system<partsize_t, particles_rnumber>>,
                       int, 1, 11, 1, // interpolation_size
                       int, 0, 3, 1, // spline_mode
                       particles_system_build_container<partsize_t, field_rnumber,be,fc,particles_rnumber>>(
                           interpolation_size, // template iterator 1
                           spline_mode, // template iterator 2
                           fs_field,fs_kk, nsteps, nparticles, fname_input, inDatanameState, inDatanameRhs, mpi_comm, in_current_iteration);
}


#endif
