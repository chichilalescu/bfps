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



#include <hdf5.h>
#include <unordered_map>
#include <vector>
#include <string>
#include "fftw_interface.hpp"
#include "field_layout.hpp"

#ifndef FIELD_HPP

#define FIELD_HPP

enum field_backend {FFTW};
enum kspace_dealias_type {TWO_THIRDS, SMOOTH};


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
        void dealias(typename fftw_interface<rnumber>::complex *__restrict__ a);

        template <typename rnumber,
                  field_components fc>
        void cospectrum(
                const rnumber(* __restrict__ a)[2],
                const rnumber(* __restrict__ b)[2],
                const hid_t group,
                const std::string dset_name,
                const hsize_t toffset);
        template <class func_type>
        void CLOOP(func_type expression)
        {
            ptrdiff_t cindex = 0;
            for (hsize_t yindex = 0; yindex < this->layout->subsizes[0]; yindex++)
            for (hsize_t zindex = 0; zindex < this->layout->subsizes[1]; zindex++)
            for (hsize_t xindex = 0; xindex < this->layout->subsizes[2]; xindex++)
                {
                    expression(cindex, xindex, yindex, zindex);
                    cindex++;
                }
        }
        template <class func_type>
        void CLOOP_K2(func_type expression)
        {
            double k2;
            ptrdiff_t cindex = 0;
            for (hsize_t yindex = 0; yindex < this->layout->subsizes[0]; yindex++)
            for (hsize_t zindex = 0; zindex < this->layout->subsizes[1]; zindex++)
            for (hsize_t xindex = 0; xindex < this->layout->subsizes[2]; xindex++)
                {
                    k2 = (this->kx[xindex]*this->kx[xindex] +
                          this->ky[yindex]*this->ky[yindex] +
                          this->kz[zindex]*this->kz[zindex]);
                    expression(cindex, xindex, yindex, zindex, k2);
                    cindex++;
                }
        }
        template <class func_type>
        void CLOOP_K2_NXMODES(func_type expression)
        {
            ptrdiff_t cindex = 0;
            for (hsize_t yindex = 0; yindex < this->layout->subsizes[0]; yindex++)
            for (hsize_t zindex = 0; zindex < this->layout->subsizes[1]; zindex++)
            {
                hsize_t xindex = 0;
                double k2 = (
                        this->kx[xindex]*this->kx[xindex] +
                        this->ky[yindex]*this->ky[yindex] +
                        this->kz[zindex]*this->kz[zindex]);
                expression(cindex, xindex, yindex, zindex, k2, 1);
                cindex++;
                for (xindex = 1; xindex < this->layout->subsizes[2]; xindex++)
                {
                    k2 = (this->kx[xindex]*this->kx[xindex] +
                          this->ky[yindex]*this->ky[yindex] +
                          this->kz[zindex]*this->kz[zindex]);
                    expression(cindex, xindex, yindex, zindex, k2, 2);
                    cindex++;
                }
            }
        }
        template <typename rnumber>
        void force_divfree(typename fftw_interface<rnumber>::complex *__restrict__ a);
};

template <typename rnumber,
          field_backend be,
          field_components fc>
class field
{
    private:
        /* data arrays */
        rnumber *__restrict__ data;
    public:
        hsize_t npoints;
        bool real_space_representation;
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
        typename fftw_interface<rnumber>::plan c2r_plan;
        typename fftw_interface<rnumber>::plan r2c_plan;
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
        void normalize();
        void symmetrize();

        void compute_rspace_xincrement_stats(
                const int xcells,
                const hid_t group,
                const std::string dset_name,
                const hsize_t toffset,
                const std::vector<double> max_estimate);

        void compute_rspace_stats(
                const hid_t group,
                const std::string dset_name,
                const hsize_t toffset,
                const std::vector<double> max_estimate);
        inline rnumber *__restrict__ get_rdata()
        {
            return this->data;
        }
        inline typename fftw_interface<rnumber>::complex *__restrict__ get_cdata()
        {
            return (typename fftw_interface<rnumber>::complex*__restrict__)this->data;
        }

        inline field<rnumber, be, fc>& operator=(const typename fftw_interface<rnumber>::complex *__restrict__ source)
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

        inline field<rnumber, be, fc>& operator=(const rnumber value)
        {
            std::fill_n(this->data,
                        this->rmemlayout->local_size,
                        value);
            return *this;
        }

        template <kspace_dealias_type dt>
        void compute_stats(
                kspace<be, dt> *kk,
                const hid_t group,
                const std::string dset_name,
                const hsize_t toffset,
                const double max_estimate);
        inline void impose_zero_mode()
        {
            if (this->clayout->myrank == this->clayout->rank[0][0] &&
                this->real_space_representation == false)
            {
                std::fill_n(this->data, 2*ncomp(fc), 0.0);
            }
        }
        template <class func_type>
        void RLOOP(func_type expression)
        {
            switch(be)
            {
                case FFTW:
                    for (hsize_t zindex = 0; zindex < this->rlayout->subsizes[0]; zindex++)
                    for (hsize_t yindex = 0; yindex < this->rlayout->subsizes[1]; yindex++)
                    {
                        ptrdiff_t rindex = (
                                zindex * this->rlayout->subsizes[1] + yindex)*(
                                    this->rmemlayout->subsizes[2]);
                        for (hsize_t xindex = 0; xindex < this->rlayout->subsizes[2]; xindex++)
                        {
                            expression(rindex, xindex, yindex, zindex);
                            rindex++;
                        }
                    }
                    break;
            }
        }
};

template <typename rnumber,
          field_backend be,
          field_components fc1,
          field_components fc2,
          kspace_dealias_type dt>
void compute_gradient(
        kspace<be, dt> *kk,
        field<rnumber, be, fc1> *source,
        field<rnumber, be, fc2> *destination);

#endif//FIELD_HPP

