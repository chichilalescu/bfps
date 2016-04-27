/**********************************************************************
*                                                                     *
*  Copyright 2015 Max Planck Institute                                *
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



#include <mpi.h>
#include <hdf5.h>
#include <fftw3-mpi.h>
#include <vector>
#include "base.hpp"

#ifndef FIELD

#define FIELD

enum field_backend {FFTW};
enum field_components {ONE, THREE, THREExTHREE};

constexpr unsigned int ncomp(
        field_components fc)
    /* return actual number of field components for each enum value */
{
    return ((fc == THREE) ? 3 : (
            (fc == THREExTHREE) ? 9 : 1));
}

constexpr unsigned int ndim(
        field_components fc)
    /* return actual number of field dimensions for each enum value */
{
    return ((fc == THREE) ? 4 : (
            (fc == THREExTHREE) ? 5 : 3));
}

template <field_components fc>
class field_layout
{
    public:
        /* description */
        hsize_t sizes[ndim(fc)];
        hsize_t subsizes[ndim(fc)];
        hsize_t starts[ndim(fc)];
        hsize_t local_size, full_size;

        int myrank, nprocs;
        MPI_Comm comm;

        std::vector<std::vector<int>> rank;
        std::vector<std::vector<int>> all_start;
        std::vector<std::vector<int>> all_size;

        /* methods */
        field_layout(
                hsize_t *SIZES,
                hsize_t *SUBSIZES,
                hsize_t *STARTS,
                MPI_Comm COMM_TO_USE);
        ~field_layout(){}
};

template <class rnumber,
          field_backend be,
          field_components fc>
class field
{
    private:
        /* data arrays */
        rnumber *data;
    public:
        /* basic MPI information */
        int myrank, nprocs;
        MPI_Comm comm;

        /* descriptions of field layout and distribution */
        field_layout<fc> *clayout, *rlayout, *rmemlayout;

        /* FFT plans */
        void *c2r_plan;
        void *r2c_plan;
        unsigned fftw_plan_rigor;

        /* HDF5 data types for arrays */
        hid_t rnumber_H5T, cnumber_H5T;

        /* methods */
        field(
                int nx, int ny, int nz,
                MPI_Comm COMM_TO_USE,
                unsigned FFTW_PLAN_RIGOR = FFTW_ESTIMATE);
        ~field();

        int io(
                const char *fname,
                const char *dset_name,
                int iteration,
                bool read = true);

        void dft();
        void ift();
};

#endif//FIELD

