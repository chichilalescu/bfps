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
#include "omputils.hpp"
#include "fftw_interface.hpp"
#include "field_layout.hpp"

#ifndef KSPACE_HPP

#define KSPACE_HPP

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
        void low_pass(
                typename fftw_interface<rnumber>::complex *__restrict__ a,
                const double kmax);

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
            #pragma omp parallel
            {
                const hsize_t start = OmpUtils::ForIntervalStart(this->layout->subsizes[0]);
                const hsize_t end = OmpUtils::ForIntervalEnd(this->layout->subsizes[0]);

                for (hsize_t yindex = start; yindex < end; yindex++){
                    ptrdiff_t cindex = yindex*this->layout->subsizes[1]*this->layout->subsizes[2];
                    for (hsize_t zindex = 0; zindex < this->layout->subsizes[1]; zindex++){
                        for (hsize_t xindex = 0; xindex < this->layout->subsizes[2]; xindex++)
                        {
                            expression(cindex, xindex, yindex, zindex);
                            cindex++;
                        }
                    }
                }
            }
        }
        template <class func_type>
        void CLOOP_K2(func_type expression)
        {
            #pragma omp parallel
            {
                const double chunk = double(this->layout->subsizes[0])/double(omp_get_num_threads());
                const hsize_t start = hsize_t(chunk*double(omp_get_thread_num()));
                const hsize_t end = (omp_get_thread_num() == omp_get_num_threads()-1) ?
                                            this->layout->subsizes[0]:
                                            hsize_t(chunk*double(omp_get_thread_num()+1));

                for (hsize_t yindex = start; yindex < end; yindex++){
                    ptrdiff_t cindex = yindex*this->layout->subsizes[1]*this->layout->subsizes[2];
                    for (hsize_t zindex = 0; zindex < this->layout->subsizes[1]; zindex++){
                        for (hsize_t xindex = 0; xindex < this->layout->subsizes[2]; xindex++)
                        {
                            double k2 = (this->kx[xindex]*this->kx[xindex] +
                                  this->ky[yindex]*this->ky[yindex] +
                                  this->kz[zindex]*this->kz[zindex]);
                            expression(cindex, xindex, yindex, zindex, k2);
                            cindex++;
                        }
                    }
                }
            }
        }
        template <class func_type>
        void CLOOP_K2_NXMODES(func_type expression)
        {
            #pragma omp parallel
            {
                const double chunk = double(this->layout->subsizes[0])/double(omp_get_num_threads());
                const hsize_t start = hsize_t(chunk*double(omp_get_thread_num()));
                const hsize_t end = (omp_get_thread_num() == omp_get_num_threads()-1) ?
                                            this->layout->subsizes[0]:
                                            hsize_t(chunk*double(omp_get_thread_num()+1));

                for (hsize_t yindex = start; yindex < end; yindex++){
                    ptrdiff_t cindex = yindex*this->layout->subsizes[1]*this->layout->subsizes[2];
                    for (hsize_t zindex = 0; zindex < this->layout->subsizes[1]; zindex++){
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
            }
        }
        template <typename rnumber>
        void force_divfree(typename fftw_interface<rnumber>::complex *__restrict__ a);
};

#endif//KSPACE_HPP

