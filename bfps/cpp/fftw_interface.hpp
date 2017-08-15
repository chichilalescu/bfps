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

#ifndef FFTW_INTERFACE_HPP
#define FFTW_INTERFACE_HPP

#include <fftw3-mpi.h>

#ifdef USE_FFTWESTIMATE
#define DEFAULT_FFTW_FLAG FFTW_ESTIMATE
#warning You are using FFTW estimate
#else
#define DEFAULT_FFTW_FLAG FFTW_PATIENT
#endif

template <class realtype>
class fftw_interface;

template <>
class fftw_interface<float>
{
public:
    using real = float;
    using complex = fftwf_complex;
    using plan = fftwf_plan;
    using iodim = fftwf_iodim;

    static complex* alloc_complex(const size_t in_size){
        return fftwf_alloc_complex(in_size);
    }

    static real* alloc_real(const size_t in_size){
        return fftwf_alloc_real(in_size);
    }

    static void free(void* ptr){
        fftwf_free(ptr);
    }

    static void execute(plan in_plan){
        fftwf_execute(in_plan);
    }

    static void destroy_plan(plan in_plan){
        fftwf_destroy_plan(in_plan);
    }

    template <class ... Params>
    static plan mpi_plan_transpose(Params ... params){
        return fftwf_mpi_plan_transpose(params...);
    }

    template <class ... Params>
    static plan mpi_plan_many_transpose(Params ... params){
        return fftwf_mpi_plan_many_transpose(params...);
    }

    template <class ... Params>
    static plan plan_guru_r2r(Params ... params){
        return fftwf_plan_guru_r2r(params...);
    }

    template <class ... Params>
    static plan plan_guru_dft(Params ... params){
        return fftwf_plan_guru_dft(params...);
    }

    template <class ... Params>
    static plan mpi_plan_many_dft_c2r(Params ... params){
        return fftwf_mpi_plan_many_dft_c2r(params...);
    }

    template <class ... Params>
    static plan mpi_plan_many_dft_r2c(Params ... params){
        return fftwf_mpi_plan_many_dft_r2c(params...);
    }

    template <class ... Params>
    static plan mpi_plan_dft_c2r_3d(Params ... params){
        return fftwf_mpi_plan_dft_c2r_3d(params...);
    }
};

template <>
class fftw_interface<double>
{
public:
    using real = double;
    using complex = fftw_complex;
    using plan = fftw_plan;
    using iodim = fftw_iodim;

    static complex* alloc_complex(const size_t in_size){
        return fftw_alloc_complex(in_size);
    }

    static real* alloc_real(const size_t in_size){
        return fftw_alloc_real(in_size);
    }

    static void free(void* ptr){
        fftw_free(ptr);
    }

    static void execute(plan in_plan){
        fftw_execute(in_plan);
    }

    static void destroy_plan(plan in_plan){
        fftw_destroy_plan(in_plan);
    }

    template <class ... Params>
    static plan mpi_plan_transpose(Params ... params){
        return fftw_mpi_plan_transpose(params...);
    }

    template <class ... Params>
    static plan mpi_plan_many_transpose(Params ... params){
        return fftw_mpi_plan_many_transpose(params...);
    }

    template <class ... Params>
    static plan plan_guru_r2r(Params ... params){
        return fftw_plan_guru_r2r(params...);
    }

    template <class ... Params>
    static plan plan_guru_dft(Params ... params){
        return fftw_plan_guru_dft(params...);
    }

    template <class ... Params>
    static plan mpi_plan_many_dft_c2r(Params ... params){
        return fftw_mpi_plan_many_dft_c2r(params...);
    }

    template <class ... Params>
    static plan mpi_plan_many_dft_r2c(Params ... params){
        return fftw_mpi_plan_many_dft_r2c(params...);
    }

    template <class ... Params>
    static plan mpi_plan_dft_c2r_3d(Params ... params){
        return fftw_mpi_plan_dft_c2r_3d(params...);
    }
};



#endif // FFTW_INTERFACE_HPP

