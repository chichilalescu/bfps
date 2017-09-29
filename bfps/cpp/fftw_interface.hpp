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

// To have multiple calls to c2r/r2c
#define SPLIT_FFTW_MANY
#ifdef SPLIT_FFTW_MANY
#include <vector>
#include <memory>
#include <algorithm>
#include <cassert>
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
#ifdef SPLIT_FFTW_MANY
    struct many_plan_container{
        int rnk;
        std::vector<ptrdiff_t> n;
        int howmany;
        ptrdiff_t iblock;
        ptrdiff_t oblock;
        std::shared_ptr<real> buffer;
        plan plan_to_use;

        ptrdiff_t local_n0, local_0_start;
        ptrdiff_t local_n1, local_1_start;

        bool is_r2c;
        void* in;
        void* out;

        ptrdiff_t nb_real_to_copy;
        ptrdiff_t nb_complex_to_copy;
    };

    using many_plan = many_plan_container;
#else
    using many_plan = fftwf_plan;
#endif

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
    static ptrdiff_t mpi_local_size_many_transposed(Params ... params){
        return fftwf_mpi_local_size_many_transposed(params...);
    }

    template <class ... Params>
    static ptrdiff_t mpi_local_size_many(Params ... params){
        return fftwf_mpi_local_size_many(params...);
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

#ifdef SPLIT_FFTW_MANY
    static many_plan mpi_plan_many_dft_c2r(int rnk, const ptrdiff_t *n, ptrdiff_t howmany,
                                                         ptrdiff_t iblock, ptrdiff_t oblock,
                                                         complex *in, real *out,
                                                         MPI_Comm comm, unsigned flags){
        assert(iblock == FFTW_MPI_DEFAULT_BLOCK);
        assert(oblock == FFTW_MPI_DEFAULT_BLOCK);

        many_plan c2r_plan;
        c2r_plan.rnk = rnk;
        c2r_plan.n.insert(c2r_plan.n.end(), n, n+rnk);
        c2r_plan.howmany = howmany;
        c2r_plan.iblock = iblock;
        c2r_plan.oblock = oblock;
        c2r_plan.is_r2c = false;
        c2r_plan.in = in;
        c2r_plan.out = out;

        // If 1 then use default without copy
        if(howmany == 1){
            c2r_plan.plan_to_use = mpi_plan_dft_c2r(rnk, n,
                                           (complex*)in,
                                           out,
                                           comm, flags);
            return c2r_plan;
        }

        // We need to find out the size of the buffer to allocate
        mpi_local_size_many_transposed(
                rnk, n, howmany,
                FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, comm,
                &c2r_plan.local_n0, &c2r_plan.local_0_start,
                &c2r_plan.local_n1, &c2r_plan.local_1_start);

        ptrdiff_t sizeBuffer = c2r_plan.local_n0/howmany;
        for(int idxrnk = 1 ; idxrnk < rnk-1 ; ++idxrnk){
            sizeBuffer *= n[idxrnk];
        }
        sizeBuffer *= n[rnk-1]+2;

        c2r_plan.buffer.reset(alloc_real(sizeBuffer));
        // Init the plan
        c2r_plan.plan_to_use = mpi_plan_dft_c2r(rnk, n,
                                         (complex*)c2r_plan.buffer.get(),
                                         c2r_plan.buffer.get(),
                                         comm, flags);

        c2r_plan.nb_real_to_copy = c2r_plan.local_n0/howmany;
        c2r_plan.nb_complex_to_copy = c2r_plan.local_n0/howmany;
        for(int idxrnk = 1 ; idxrnk < rnk-1 ; ++idxrnk){
            c2r_plan.nb_real_to_copy *= n[idxrnk];
            c2r_plan.nb_complex_to_copy *= n[idxrnk];
        }
        c2r_plan.nb_real_to_copy *= n[rnk-1];
        c2r_plan.nb_complex_to_copy *= n[rnk-1]/2 + 1;

        return c2r_plan;
    }

    static many_plan mpi_plan_many_dft_r2c(int rnk, const ptrdiff_t *n, ptrdiff_t howmany,
                                                         ptrdiff_t iblock, ptrdiff_t oblock,
                                                         real *in, complex *out,
                                                         MPI_Comm comm, unsigned flags){
        assert(iblock == FFTW_MPI_DEFAULT_BLOCK);
        assert(oblock == FFTW_MPI_DEFAULT_BLOCK);

        many_plan r2c_plan;
        r2c_plan.rnk = rnk;
        r2c_plan.n.insert(r2c_plan.n.end(), n, n+rnk);
        r2c_plan.howmany = howmany;
        r2c_plan.iblock = iblock;
        r2c_plan.oblock = oblock;
        r2c_plan.is_r2c = true;
        r2c_plan.in = in;
        r2c_plan.out = out;

        // If 1 then use default without copy
        if(howmany == 1){
            r2c_plan.plan_to_use = mpi_plan_dft_r2c(rnk, n,
                                           in,
                                           (complex*)out,
                                           comm, flags);
            return r2c_plan;
        }

        // We need to find out the size of the buffer to allocate
        mpi_local_size_many_transposed(
                rnk, n, howmany,
                FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, comm,
                &r2c_plan.local_n0, &r2c_plan.local_0_start,
                &r2c_plan.local_n1, &r2c_plan.local_1_start);

        ptrdiff_t sizeBuffer = r2c_plan.local_n0/howmany;
        for(int idxrnk = 1 ; idxrnk < rnk-1 ; ++idxrnk){
            sizeBuffer *= n[idxrnk];
        }
        sizeBuffer *= n[rnk-1]+2;

        r2c_plan.buffer.reset(alloc_real(sizeBuffer));
        // Init the plan
        r2c_plan.plan_to_use = mpi_plan_dft_r2c(rnk, n,
                                         r2c_plan.buffer.get(),
                                         (complex*)r2c_plan.buffer.get(),
                                         comm, flags);


        r2c_plan.nb_real_to_copy = r2c_plan.local_n0/howmany;
        r2c_plan.nb_complex_to_copy = r2c_plan.local_n0/howmany;
        for(int idxrnk = 1 ; idxrnk < rnk-1 ; ++idxrnk){
            r2c_plan.nb_real_to_copy *= n[idxrnk];
            r2c_plan.nb_complex_to_copy *= n[idxrnk];
        }
        r2c_plan.nb_real_to_copy *= n[rnk-1];
        r2c_plan.nb_complex_to_copy *= n[rnk-1]/2 + 1;

        return r2c_plan;
    }

    static void execute(many_plan& in_plan){
        if(in_plan.howmany == 1){
            execute(in_plan.plan_to_use);
            return;
        }

        const ptrdiff_t nb_to_copy_to_buffer = (in_plan.is_r2c?in_plan.nb_real_to_copy:in_plan.nb_complex_to_copy);
        const ptrdiff_t nb_to_copy_from_buffer = (!in_plan.is_r2c?in_plan.nb_real_to_copy:in_plan.nb_complex_to_copy);

        for(int idx_howmany = 0 ; idx_howmany < in_plan.howmany ; ++idx_howmany){
            // Copy to buffer
            if(in_plan.is_r2c){
                real* dest = in_plan.buffer.get();
                const real* src = ((const real*)in_plan.in)+idx_howmany;
                for(ptrdiff_t idx_copy = 0 ; idx_copy < nb_to_copy_to_buffer ; ++idx_copy){
                    dest[idx_copy] = src[idx_copy*in_plan.howmany];
                }
            }
            else{
                complex* dest = (complex*)in_plan.buffer.get();
                const complex* src = ((const complex*)in_plan.in)+idx_howmany;
                for(ptrdiff_t idx_copy = 0 ; idx_copy < nb_to_copy_to_buffer ; ++idx_copy){
                    dest[idx_copy][0] = src[idx_copy*in_plan.howmany][0];
                    dest[idx_copy][1] = src[idx_copy*in_plan.howmany][1];
                }
            }

            execute(in_plan.plan_to_use);
            // Copy result from buffer
            if(in_plan.is_r2c){
                complex* dest = ((complex*)in_plan.in)+idx_howmany;
                const complex* src = (const complex*)in_plan.buffer.get();
                for(ptrdiff_t idx_copy = 0 ; idx_copy < nb_to_copy_from_buffer ; ++idx_copy){
                    dest[idx_copy*in_plan.howmany][0] = src[idx_copy][0];
                    dest[idx_copy*in_plan.howmany][1] = src[idx_copy][1];
                }
            }
            else{
                real* dest = ((real*)in_plan.in)+idx_howmany;
                const real* src = in_plan.buffer.get();
                for(ptrdiff_t idx_copy = 0 ; idx_copy < nb_to_copy_from_buffer ; ++idx_copy){
                    dest[idx_copy*in_plan.howmany] = src[idx_copy];
                }
            }
        }
    }

    static void destroy_plan(many_plan& in_plan){
        destroy_plan(in_plan.plan_to_use);
    }
#else
    template <class ... Params>
    static plan mpi_plan_many_dft_c2r(Params ... params){
        return fftwf_mpi_plan_many_dft_c2r(params...);
    }

    template <class ... Params>
    static plan mpi_plan_many_dft_r2c(Params ... params){
        return fftwf_mpi_plan_many_dft_r2c(params...);
    }
#endif

    template <class ... Params>
    static plan mpi_plan_dft_c2r(Params ... params){
        return fftwf_mpi_plan_dft_c2r(params...);
    }

    template <class ... Params>
    static plan mpi_plan_dft_r2c(Params ... params){
        return fftwf_mpi_plan_dft_r2c(params...);
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
#ifdef SPLIT_FFTW_MANY
    struct many_plan_container{
        int rnk;
        std::vector<ptrdiff_t> n;
        int howmany;
        ptrdiff_t iblock;
        ptrdiff_t oblock;
        std::shared_ptr<real> buffer;
        plan plan_to_use;

        ptrdiff_t local_n0, local_0_start;
        ptrdiff_t local_n1, local_1_start;

        bool is_r2c;
        void* in;
        void* out;

        ptrdiff_t nb_real_to_copy;
        ptrdiff_t nb_complex_to_copy;
    };

    using many_plan = many_plan_container;
#else
    using many_plan = fftw_plan;
#endif

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
    static ptrdiff_t mpi_local_size_many_transposed(Params ... params){
        return fftw_mpi_local_size_many_transposed(params...);
    }

    template <class ... Params>
    static ptrdiff_t mpi_local_size_many(Params ... params){
        return fftw_mpi_local_size_many(params...);
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


#ifdef SPLIT_FFTW_MANY
    static many_plan mpi_plan_many_dft_c2r(int rnk, const ptrdiff_t *n, ptrdiff_t howmany,
                                                         ptrdiff_t iblock, ptrdiff_t oblock,
                                                         complex *in, real *out,
                                                         MPI_Comm comm, unsigned flags){
        assert(iblock == FFTW_MPI_DEFAULT_BLOCK);
        assert(oblock == FFTW_MPI_DEFAULT_BLOCK);

        many_plan c2r_plan;
        c2r_plan.rnk = rnk;
        c2r_plan.n.insert(c2r_plan.n.end(), n, n+rnk);
        c2r_plan.howmany = howmany;
        c2r_plan.iblock = iblock;
        c2r_plan.oblock = oblock;
        c2r_plan.is_r2c = false;
        c2r_plan.in = in;
        c2r_plan.out = out;

        // If 1 then use default without copy
        if(howmany == 1){
            c2r_plan.plan_to_use = mpi_plan_dft_c2r(rnk, n,
                                           (complex*)in,
                                           out,
                                           comm, flags);
            return c2r_plan;
        }

        // We need to find out the size of the buffer to allocate
        mpi_local_size_many_transposed(
                rnk, n, howmany,
                FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, comm,
                &c2r_plan.local_n0, &c2r_plan.local_0_start,
                &c2r_plan.local_n1, &c2r_plan.local_1_start);

        ptrdiff_t sizeBuffer = c2r_plan.local_n0/howmany;
        for(int idxrnk = 1 ; idxrnk < rnk-1 ; ++idxrnk){
            sizeBuffer *= n[idxrnk];
        }
        sizeBuffer *= n[rnk-1]+2;

        c2r_plan.buffer.reset(alloc_real(sizeBuffer));
        // Init the plan
        c2r_plan.plan_to_use = mpi_plan_dft_c2r(rnk, n,
                                         (complex*)c2r_plan.buffer.get(),
                                         c2r_plan.buffer.get(),
                                         comm, flags);

        c2r_plan.nb_real_to_copy = c2r_plan.local_n0/howmany;
        c2r_plan.nb_complex_to_copy = c2r_plan.local_n0/howmany;
        for(int idxrnk = 1 ; idxrnk < rnk-1 ; ++idxrnk){
            c2r_plan.nb_real_to_copy *= n[idxrnk];
            c2r_plan.nb_complex_to_copy *= n[idxrnk];
        }
        c2r_plan.nb_real_to_copy *= n[rnk-1];
        c2r_plan.nb_complex_to_copy *= n[rnk-1]/2 + 1;

        return c2r_plan;
    }

    static many_plan mpi_plan_many_dft_r2c(int rnk, const ptrdiff_t *n, ptrdiff_t howmany,
                                                         ptrdiff_t iblock, ptrdiff_t oblock,
                                                         real *in, complex *out,
                                                         MPI_Comm comm, unsigned flags){
        assert(iblock == FFTW_MPI_DEFAULT_BLOCK);
        assert(oblock == FFTW_MPI_DEFAULT_BLOCK);

        many_plan r2c_plan;
        r2c_plan.rnk = rnk;
        r2c_plan.n.insert(r2c_plan.n.end(), n, n+rnk);
        r2c_plan.howmany = howmany;
        r2c_plan.iblock = iblock;
        r2c_plan.oblock = oblock;
        r2c_plan.is_r2c = true;
        r2c_plan.in = in;
        r2c_plan.out = out;

        // If 1 then use default without copy
        if(howmany == 1){
            r2c_plan.plan_to_use = mpi_plan_dft_r2c(rnk, n,
                                           in,
                                           (complex*)out,
                                           comm, flags);
            return r2c_plan;
        }

        // We need to find out the size of the buffer to allocate
        mpi_local_size_many_transposed(
                rnk, n, howmany,
                FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, comm,
                &r2c_plan.local_n0, &r2c_plan.local_0_start,
                &r2c_plan.local_n1, &r2c_plan.local_1_start);

        ptrdiff_t sizeBuffer = r2c_plan.local_n0/howmany;
        for(int idxrnk = 1 ; idxrnk < rnk-1 ; ++idxrnk){
            sizeBuffer *= n[idxrnk];
        }
        sizeBuffer *= n[rnk-1]+2;

        r2c_plan.buffer.reset(alloc_real(sizeBuffer));
        // Init the plan
        r2c_plan.plan_to_use = mpi_plan_dft_r2c(rnk, n,
                                         r2c_plan.buffer.get(),
                                         (complex*)r2c_plan.buffer.get(),
                                         comm, flags);


        r2c_plan.nb_real_to_copy = r2c_plan.local_n0/howmany;
        r2c_plan.nb_complex_to_copy = r2c_plan.local_n0/howmany;
        for(int idxrnk = 1 ; idxrnk < rnk-1 ; ++idxrnk){
            r2c_plan.nb_real_to_copy *= n[idxrnk];
            r2c_plan.nb_complex_to_copy *= n[idxrnk];
        }
        r2c_plan.nb_real_to_copy *= n[rnk-1];
        r2c_plan.nb_complex_to_copy *= n[rnk-1]/2 + 1;

        return r2c_plan;
    }

    static void execute(many_plan& in_plan){
        if(in_plan.howmany == 1){
            execute(in_plan.plan_to_use);
            return;
        }

        const ptrdiff_t nb_to_copy_to_buffer = (in_plan.is_r2c?in_plan.nb_real_to_copy:in_plan.nb_complex_to_copy);
        const ptrdiff_t nb_to_copy_from_buffer = (!in_plan.is_r2c?in_plan.nb_real_to_copy:in_plan.nb_complex_to_copy);

        for(int idx_howmany = 0 ; idx_howmany < in_plan.howmany ; ++idx_howmany){
            // Copy to buffer
            if(in_plan.is_r2c){
                real* dest = in_plan.buffer.get();
                const real* src = ((const real*)in_plan.in)+idx_howmany;
                for(ptrdiff_t idx_copy = 0 ; idx_copy < nb_to_copy_to_buffer ; ++idx_copy){
                    dest[idx_copy] = src[idx_copy*in_plan.howmany];
                }
            }
            else{
                complex* dest = (complex*)in_plan.buffer.get();
                const complex* src = ((const complex*)in_plan.in)+idx_howmany;
                for(ptrdiff_t idx_copy = 0 ; idx_copy < nb_to_copy_to_buffer ; ++idx_copy){
                    dest[idx_copy][0] = src[idx_copy*in_plan.howmany][0];
                    dest[idx_copy][1] = src[idx_copy*in_plan.howmany][1];
                }
            }

            execute(in_plan.plan_to_use);
            // Copy result from buffer
            if(in_plan.is_r2c){
                complex* dest = ((complex*)in_plan.in)+idx_howmany;
                const complex* src = (const complex*)in_plan.buffer.get();
                for(ptrdiff_t idx_copy = 0 ; idx_copy < nb_to_copy_from_buffer ; ++idx_copy){
                    dest[idx_copy*in_plan.howmany][0] = src[idx_copy][0];
                    dest[idx_copy*in_plan.howmany][1] = src[idx_copy][1];
                }
            }
            else{
                real* dest = ((real*)in_plan.in)+idx_howmany;
                const real* src = in_plan.buffer.get();
                for(ptrdiff_t idx_copy = 0 ; idx_copy < nb_to_copy_from_buffer ; ++idx_copy){
                    dest[idx_copy*in_plan.howmany] = src[idx_copy];
                }
            }
        }
    }

    static void destroy_plan(many_plan& in_plan){
        destroy_plan(in_plan.plan_to_use);
    }
#else
    template <class ... Params>
    static plan mpi_plan_many_dft_c2r(Params ... params){
        return fftw_mpi_plan_many_dft_c2r(params...);
    }

    template <class ... Params>
    static plan mpi_plan_many_dft_r2c(Params ... params){
        return fftw_mpi_plan_many_dft_r2c(params...);
    }
#endif

    template <class ... Params>
    static plan mpi_plan_dft_c2r(Params ... params){
        return fftw_mpi_plan_dft_c2r(params...);
    }

    template <class ... Params>
    static plan mpi_plan_dft_r2c(Params ... params){
        return fftw_mpi_plan_dft_r2c(params...);
    }

    template <class ... Params>
    static plan mpi_plan_dft_c2r_3d(Params ... params){
        return fftw_mpi_plan_dft_c2r_3d(params...);
    }
};



#endif // FFTW_INTERFACE_HPP

