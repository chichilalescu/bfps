/**********************************************************************
*                                                                     *
*  Copyright 2017 Max Planck Institute                                *
*                 for Dynamics and Self-Organization                  *
*                                                                     *
*  This file is part of bfps.                                         *
*                                                                     *
*  bfps is free software: you can redistribute it and/or modify       *
*  it under the terms of the GNU General Public License as published  *
*  by the Free Software Foundation, either version 3 of the License,  *
*  or (at your option) any later version.                             *
*                                                                     *
*  bfps is distributed in the hope that it will be useful,            *
*  but WITHOUT ANY WARRANTY; without even the implied warranty of     *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *
*  GNU General Public License for more details.                       *
*                                                                     *
*  You should have received a copy of the GNU General Public License  *
*  along with bfps.  If not, see <http://www.gnu.org/licenses/>       *
*                                                                     *
* Contact: Cristian.Lalescu@ds.mpg.de                                 *
*                                                                     *
**********************************************************************/



#ifndef NATIVE_BINARY_TO_HDF5_HPP
#define NATIVE_BINARY_TO_HDF5_HPP

#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include "base.hpp"
#include "field.hpp"
#include "field_binary_IO.hpp"
#include "full_code/postprocess.hpp"

template <typename rnumber>
class native_binary_to_hdf5: public postprocess
{
    public:

        field<rnumber, FFTW, THREE> *vec_field;
        field_binary_IO<rnumber, COMPLEX, THREE> *bin_IO;

        native_binary_to_hdf5(
                const MPI_Comm COMMUNICATOR,
                const std::string &simulation_name):
            postprocess(
                    COMMUNICATOR,
                    simulation_name){}
        virtual ~native_binary_to_hdf5(){}

        int initialize(void);
        int work_on_current_iteration(void);
        int finalize(void);
        virtual int read_parameters(void);
};

#endif//NATIVE_BINARY_TO_HDF5_HPP

