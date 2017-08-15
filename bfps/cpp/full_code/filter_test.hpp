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

/** \brief A class for testing filters.
 *
 *  This class applies available filters to three Dirac distributions:
 *      - nonzero at the origin
 *      - nonzero on the `x` axis
 *      - nonzero on the `(x, y)` plane.
 *  All three distributions are normalized, so simple sanity checks can
 *  be performed afterwards in a Python script.
 *
 *  While the convolutions can obviously be implemented in Python directly,
 *  it's better if the functionality is available here directly for easy
 *  reference.
 */

template <typename rnumber>
class filter_test: public test
{
    public:

        /* parameters that are read in read_parameters */
        double filter_length;

        /* other stuff */
        kspace<FFTW, SMOOTH> *kk;
        field<rnumber, FFTW, ONE> *scal_field;

        filter_test(
                const MPI_Comm COMMUNICATOR,
                const std::string &simulation_name):
            test(
                    COMMUNICATOR,
                    simulation_name){}
        ~filter_test(){}

        int initialize(void);
        int do_work(void);
        int finalize(void);
        int read_parameters(void);

        int reset_field(int dimension);
};

#endif//FILTER_TEST_HPP

