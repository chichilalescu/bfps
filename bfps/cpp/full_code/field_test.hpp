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



#ifndef FILTER_TEST_HPP
#define FILTER_TEST_HPP



#include <cstdlib>
#include "base.hpp"
#include "kspace.hpp"
#include "field.hpp"
#include "full_code/test.hpp"

/** \brief A class for testing basic field class functionality.
 */

template <typename rnumber>
class field_test: public test
{
    public:
        double filter_length;
        // kspace, in case we want to compute spectra or smth

        field_test(
                const MPI_Comm COMMUNICATOR,
                const std::string &simulation_name):
            test(
                    COMMUNICATOR,
                    simulation_name){}
        ~field_test(){}

        int initialize(void);
        int do_work(void);
        int finalize(void);
        int read_parameters(void);
};

#endif//FILTER_TEST_HPP

