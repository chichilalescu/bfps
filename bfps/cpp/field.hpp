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
#include <unordered_map>
#include <vector>
#include <string>
#include "base.hpp"

#ifndef FIELD

#define FIELD

enum field_backend {FFTW};
enum field_components {ONE, THREE, THREExTHREE};
enum kspace_dealias_type {TWO_THIRDS, SMOOTH};

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
                const hsize_t *SIZES,
                const hsize_t *SUBSIZES,
                const hsize_t *STARTS,
                const MPI_Comm COMM_TO_USE);
        ~field_layout(){}
};

template <field_backend be,
          kspace_dealias_type dt>
class kspace
{
    public:
        /* relevant field layout */
        field_layout<ONE> *layout;

        /* physical parameters */
        double dkx, dky, dkz, dk, dk2;

        /* mode and dealiasing information */
        double kMx, kMy, kMz, kM, kM2;
        double kMspec, kMspec2;
        std::vector<double> kx, ky, kz;
        std::unordered_map<int, double> dealias_filter;
        std::vector<double> kshell;
        std::vector<int64_t> nshell;
        int nshells;

        /* methods */
        template <field_components fc>
        kspace(
                const field_layout<fc> *source_layout,
                const double DKX = 1.0,
                const double DKY = 1.0,
                const double DKZ = 1.0);
        ~kspace();

        template <typename rnumber,
                  field_components fc>
        void low_pass(rnumber *__restrict__ a, const double kmax);

        template <typename rnumber,
                  field_components fc>
        void dealias(rnumber *__restrict__ a);

        template <typename rnumber,
                  field_components fc>
        void cospectrum(
                const rnumber(* __restrict__ a)[2],
                const rnumber(* __restrict__ b)[2],
                const hid_t group,
                const std::string dset_name,
                const hsize_t toffset);
};

template <typename rnumber,
          field_backend be,
          field_components fc>
class field
{
    private:
        /* data arrays */
        rnumber *data;
        bool real_space_representation;
        typedef rnumber cnumber[2];
        hsize_t npoints;
    public:
        /* basic MPI information */
        int myrank, nprocs;
        MPI_Comm comm;

        /* descriptions of field layout and distribution */
        /* for the FFTW backend, at least, the real space field requires more
         * space to be allocated than strictly needed for the data, hence the
         * two layout descriptors.
         * */
        field_layout<fc> *clayout, *rlayout, *rmemlayout;

        /* FFT plans */
        void *c2r_plan;
        void *r2c_plan;
        unsigned fftw_plan_rigor;

        /* HDF5 data types for arrays */
        hid_t rnumber_H5T, cnumber_H5T;

        /* methods */
        field(
                const int nx,
                const int ny,
                const int nz,
                const MPI_Comm COMM_TO_USE,
                const unsigned FFTW_PLAN_RIGOR = FFTW_ESTIMATE);
        ~field();

        int io(
                const std::string fname,
                const std::string dset_name,
                const int toffset,
                const bool read = true);

        void dft();
        void ift();

        void compute_rspace_stats(
                const hid_t group,
                const std::string dset_name,
                const hsize_t toffset,
                const std::vector<double> max_estimate);
        inline rnumber *__restrict__ get_rdata()
        {
            return this->data;
        }
        inline cnumber *__restrict__ get_cdata()
        {
            return (cnumber*)this->data;
        }

        inline field<rnumber, be, fc>& operator=(const cnumber *__restrict__ source)
        {
            std::copy((rnumber*)source,
                      (rnumber*)(source + this->clayout->local_size),
                      this->data);
            this->real_space_representation = false;
            return *this;
        }

        inline field<rnumber, be, fc>& operator=(const rnumber *__restrict__ source)
        {
            std::copy(source,
                      source + this->rmemlayout->local_size,
                      this->data);
            this->real_space_representation = true;
            return *this;
        }
        template <kspace_dealias_type dt>
        void compute_stats(
                kspace<be, dt> *kk,
                const hid_t group,
                const std::string dset_name,
                const hsize_t toffset,
                const double max_estimate);
};

/* real space loop */
#define FIELD_RLOOP(obj, expression) \
 \
{ \
    switch (be) \
    { \
        case FFTW: \
            for (hsize_t zindex = 0; zindex < obj->rlayout->subsizes[0]; zindex++) \
            for (hsize_t yindex = 0; yindex < obj->rlayout->subsizes[1]; yindex++) \
            { \
                ptrdiff_t rindex = ( \
                        zindex * obj->rlayout->subsizes[1] + yindex)*( \
                            obj->rmemlayout->subsizes[2]); \
            for (hsize_t xindex = 0; xindex < obj->rlayout->subsizes[2]; xindex++) \
                { \
                    expression; \
                    rindex++; \
                } \
            } \
            break; \
    } \
}

#define KSPACE_CLOOP_K2(obj, expression) \
 \
{ \
    double k2; \
    ptrdiff_t cindex = 0; \
    for (hsize_t yindex = 0; yindex < obj->layout->subsizes[0]; yindex++) \
    for (hsize_t zindex = 0; zindex < obj->layout->subsizes[1]; zindex++) \
    for (hsize_t xindex = 0; xindex < obj->layout->subsizes[2]; xindex++) \
        { \
            k2 = (obj->kx[xindex]*obj->kx[xindex] + \
                  obj->ky[yindex]*obj->ky[yindex] + \
                  obj->kz[zindex]*obj->kz[zindex]); \
            expression; \
            cindex++; \
        } \
}

#define KSPACE_CLOOP_K2_NXMODES(obj, expression) \
 \
{ \
    double k2; \
    ptrdiff_t cindex = 0; \
    for (hsize_t yindex = 0; yindex < obj->layout->subsizes[0]; yindex++) \
    for (hsize_t zindex = 0; zindex < obj->layout->subsizes[1]; zindex++) \
    { \
        int nxmodes = 1; \
        hsize_t xindex = 0; \
        k2 = (obj->kx[xindex]*obj->kx[xindex] + \
              obj->ky[yindex]*obj->ky[yindex] + \
              obj->kz[zindex]*obj->kz[zindex]); \
        expression; \
        cindex++; \
        nxmodes = 2; \
    for (xindex = 1; xindex < obj->layout->subsizes[2]; xindex++) \
        { \
            k2 = (obj->kx[xindex]*obj->kx[xindex] + \
                  obj->ky[yindex]*obj->ky[yindex] + \
                  obj->kz[zindex]*obj->kz[zindex]); \
            expression; \
            cindex++; \
        } \
    } \
}

#endif//FIELD

