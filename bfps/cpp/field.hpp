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
#include "kspace.hpp"
#include "omputils.hpp"

#ifndef FIELD_HPP

#define FIELD_HPP

/** \class field
 *  \brief Holds field data, performs FFTs and HDF5 I/O operations.
 *
 *  The purpose of this class is to manage memory for field data, create/destroy
 *  FFT plans for them, and compute HDF5 input/output operations.
 *
 *  FFTW recommendations are to create different plans for different arrays,
 *  hence the plans are member variables.
 *  All plans are for in-place transforms, since even with out-of-place transforms
 *  there are no guarantees that input data is not messed up by an inverse FFT, so
 *  there's no point in wasting the memory.
 *
 *
 */

template <typename rnumber,
          field_backend be,
          field_components fc>
class field
{
    private:
        rnumber *__restrict__ data; /**< data array */
    public:
        hsize_t npoints; /**< total number of grid points. Useful for normalization. */
        bool real_space_representation; /**< `true` if field is in real space representation. */

        int myrank, nprocs; /**< basic MPI information. */
        MPI_Comm comm;      /**< MPI communicator this fields lives in. */

        /* descriptions of field layout and distribution */
        /* for the FFTW backend, at least, the real space field requires more
         * space to be allocated than strictly needed for the data, hence the
         * two real space layout descriptors.
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
                const unsigned FFTW_PLAN_RIGOR = DEFAULT_FFTW_FLAG);
        ~field();

        int io(
                const std::string fname,
                const std::string field_name,
                const int iteration,
                const bool read = true);
        int io_database(
                const std::string fname,
                const std::string field_name,
                const int toffset,
                const bool read = true);

        int write_0slice(
                const hid_t group,
                const std::string field_name,
                const int iteration);

        int io_binary(
                const std::string fname,
                const int iteration,
                const bool read = true);

        /* essential FFT stuff */
        void dft();
        void ift();
        void normalize();
        void symmetrize();

        /* stats */
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

        /* acess data */
        inline rnumber *__restrict__ get_rdata()
        {
            return this->data;
        }

        inline const rnumber *__restrict__ get_rdata() const
        {
            return this->data;
        }

        inline typename fftw_interface<rnumber>::complex *__restrict__ get_cdata()
        {
            return (typename fftw_interface<rnumber>::complex*__restrict__)this->data;
        }

        inline rnumber &rval(ptrdiff_t rindex, unsigned int component = 0)
        {
            assert(fc == ONE || fc == THREE);
            assert(component >= 0 && component < ncomp(fc));
            return *(this->data + rindex*ncomp(fc) + component);
        }

        inline const rnumber& rval(ptrdiff_t rindex, unsigned int component = 0) const
        {
            assert(fc == ONE || fc == THREE);
            assert(component >= 0 && component < ncomp(fc));
            return *(this->data + rindex*ncomp(fc) + component);
        }

        inline rnumber &rval(ptrdiff_t rindex, int comp1, int comp0)
        {
            assert(fc == THREExTHREE);
            assert(comp1 >= 0 && comp1 < 3);
            assert(comp0 >= 0 && comp0 < 3);
            return *(this->data + ((rindex*3 + comp1)*3 + comp0));
        }

        inline rnumber &cval(ptrdiff_t cindex, int imag)
        {
            assert(fc == ONE);
            assert(imag == 0 || imag == 1);
            return *(this->data + cindex*2 + imag);
        }

        inline rnumber &cval(ptrdiff_t cindex, int component, int imag)
        {
            assert(fc == THREE);
            assert(imag == 0 || imag == 1);
            return *(this->data + (cindex*ncomp(fc) + component)*2 + imag);
        }

        inline rnumber &cval(ptrdiff_t cindex, int comp1, int comp0, int imag)
        {
            assert(fc == THREExTHREE);
            assert(comp1 >= 0 && comp1 < 3);
            assert(comp0 >= 0 && comp0 < 3);
            assert(imag == 0 || imag == 1);
            return *(this->data + ((cindex*3 + comp1)*3+comp0)*2 + imag);
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
                    #pragma omp parallel
                    {
                        const hsize_t start = OmpUtils::ForIntervalStart(this->rlayout->subsizes[1]);
                        const hsize_t end = OmpUtils::ForIntervalEnd(this->rlayout->subsizes[1]);

                        for (hsize_t zindex = 0; zindex < this->rlayout->subsizes[0]; zindex++)
                        for (hsize_t yindex = start; yindex < end; yindex++)
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
                    }
                    break;
            }
        }
        ptrdiff_t get_cindex(
                ptrdiff_t xindex,
                ptrdiff_t yindex,
                ptrdiff_t zindex)
        {
            return ((yindex*this->clayout->subsizes[1] +
                     zindex)*this->clayout->subsizes[2] +
                    xindex);
        }

        ptrdiff_t get_rindex(
                ptrdiff_t xindex,
                ptrdiff_t yindex,
                ptrdiff_t zindex) const
        {
            return ((zindex*this->rmemlayout->subsizes[1] +
                     yindex)*this->rmemlayout->subsizes[2] +
                    xindex);
        }

        ptrdiff_t get_rindex_from_global(const ptrdiff_t in_global_x, const ptrdiff_t in_global_y, const ptrdiff_t in_global_z) const {
            assert(in_global_x >= 0 && in_global_x < ptrdiff_t(this->rlayout->sizes[2]));
            assert(in_global_y >= 0 && in_global_y < ptrdiff_t(this->rlayout->sizes[1]));
            assert(in_global_z >= 0 && in_global_z < ptrdiff_t(this->rlayout->sizes[0]));
            return get_rindex(in_global_x - this->rlayout->starts[2],
                              in_global_y - this->rlayout->starts[1],
                              in_global_z - this->rlayout->starts[0]);
        }
};

template <typename rnumber,
          field_backend be,
          field_components fc1,
          field_components fc2,
          kspace_dealias_type dt>
int compute_gradient(
        kspace<be, dt> *kk,
        field<rnumber, be, fc1> *source,
        field<rnumber, be, fc2> *destination);

template <typename rnumber,
          field_backend be,
          kspace_dealias_type dt>
int invert_curl(
        kspace<be, dt> *kk,
        field<rnumber, be, THREE> *source,
        field<rnumber, be, THREE> *destination);

template <typename rnumber,
          field_backend be,
          field_components fc>
int joint_rspace_PDF(
        field<rnumber, be, fc> *f1,
        field<rnumber, be, fc> *f2,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset,
        const std::vector<double> max_f1_estimate,
        const std::vector<double> max_f2_estimate);

#endif//FIELD_HPP

