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

// code is generally compiled via setuptools, therefore NDEBUG is present
//#ifdef NDEBUG
//#undef NDEBUG
//#endif//NDEBUG

#include <cassert>
#include <cmath>
#include <cstring>
#include "base.hpp"
#include "fluid_solver_base.hpp"
#include "fftw_tools.hpp"


template <class rnumber>
void fluid_solver_base<rnumber>::fill_up_filename(const char *base_name, char *destination)
{
    sprintf(destination, "%s_%s_i%.5x", this->name, base_name, this->iteration);
}

template <class rnumber>
void fluid_solver_base<rnumber>::clean_up_real_space(rnumber *a, int howmany)
{
    for (ptrdiff_t rindex = 0; rindex < this->cd->local_size*2; rindex += howmany*(this->rd->subsizes[2]+2))
        std::fill_n(a+rindex+this->rd->subsizes[2]*howmany, 2*howmany, 0.0);
}

template <class rnumber>
rnumber fluid_solver_base<rnumber>::autocorrel(cnumber *a)
{
    double *spec = fftw_alloc_real(this->nshells*9);
    double sum_local, sum;
    this->cospectrum(a, a, spec);
    sum_local = 0.0;
    for (int n = 0; n < this->nshells; n++)
        sum_local += spec[n*9] + spec[n*9 + 4] + spec[n*9 + 8];
    fftw_free(spec);
    MPI_Allreduce(
            &sum_local,
            &sum,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            this->cd->comm);
    return sum;
}

template <class rnumber>
void fluid_solver_base<rnumber>::cospectrum(cnumber *a, cnumber *b, double *spec)
{
    double *cospec_local = fftw_alloc_real(this->nshells*9);
    std::fill_n(cospec_local, this->nshells*9, 0);
    int tmp_int;
    CLOOP_K2_NXMODES(
            if (k2 <= this->kM2)
            {
                tmp_int = int(sqrt(k2)/this->dk)*9;
                for (int i=0; i<3; i++)
                    for (int j=0; j<3; j++)
                    {
                        cospec_local[tmp_int+i*3+j] += nxmodes * (
                        (*(a + 3*cindex+i))[0] * (*(b + 3*cindex+j))[0] +
                        (*(a + 3*cindex+i))[1] * (*(b + 3*cindex+j))[1]);
                    }
            }
            );
    MPI_Allreduce(
            (void*)cospec_local,
            (void*)spec,
            this->nshells*9,
            MPI_DOUBLE, MPI_SUM, this->cd->comm);
    fftw_free(cospec_local);
}

template <class rnumber>
void fluid_solver_base<rnumber>::cospectrum(cnumber *a, cnumber *b, double *spec, const double k2exponent)
{
    double *cospec_local = fftw_alloc_real(this->nshells*9);
    std::fill_n(cospec_local, this->nshells*9, 0);
    double factor = 1;
    int tmp_int;
    CLOOP_K2_NXMODES(
            if (k2 <= this->kM2)
            {
                factor = nxmodes*pow(k2, k2exponent);
                tmp_int = int(sqrt(k2)/this->dk)*9;
                for (int i=0; i<3; i++)
                    for (int j=0; j<3; j++)
                    {
                        cospec_local[tmp_int+i*3+j] += factor * (
                        (*(a + 3*cindex+i))[0] * (*(b + 3*cindex+j))[0] +
                        (*(a + 3*cindex+i))[1] * (*(b + 3*cindex+j))[1]);
                    }
            }
            );
    MPI_Allreduce(
            (void*)cospec_local,
            (void*)spec,
            this->nshells*9,
            MPI_DOUBLE, MPI_SUM, this->cd->comm);
    //for (int n=0; n<this->nshells; n++)
    //{
    //    spec[n] *= 12.5663706144*pow(this->kshell[n], 2) / this->nshell[n];
    //    /*is normalization needed?
    //     * spec[n] /= this->normalization_factor*/
    //}
    fftw_free(cospec_local);
}

template <class rnumber>
void fluid_solver_base<rnumber>::compute_rspace_stats(
        rnumber *a,
        double *moments,
        ptrdiff_t *hist,
        double max_estimate,
        int nbins)
{
    double *local_moments = fftw_alloc_real(10*4);
    double val_tmp[4], binsize, pow_tmp[4];
    ptrdiff_t *local_hist = new ptrdiff_t[nbins*4];
    int bin;
    binsize = 2*max_estimate / nbins;
    std::fill_n(local_hist, nbins*4, 0);
    std::fill_n(local_moments, 10*4, 0);
    local_moments[3] = max_estimate;
    RLOOP(
        this,
        std::fill_n(pow_tmp, 4, 1.0);
        val_tmp[3] = 0.0;
        for (int i=0; i<3; i++)
        {
            val_tmp[i] = a[rindex*3+i];
            val_tmp[3] += val_tmp[i]*val_tmp[i];
        }
        val_tmp[3] = sqrt(val_tmp[3]);
        if (val_tmp[3] < local_moments[0*4+3])
            local_moments[0*4+3] = val_tmp[3];
        if (val_tmp[3] > local_moments[9*4+3])
            local_moments[9*4+3] = val_tmp[3];
        bin = int(val_tmp[3]*2/binsize);
        if (bin >= 0 && bin < nbins)
            local_hist[bin*4+3]++;
        for (int i=0; i<3; i++)
        {
            if (val_tmp[i] < local_moments[0*4+i])
                local_moments[0*4+i] = val_tmp[i];
            if (val_tmp[i] > local_moments[9*4+i])
                local_moments[9*4+i] = val_tmp[i];
            bin = int((val_tmp[i] + max_estimate) / binsize);
            if (bin >= 0 && bin < nbins)
                local_hist[bin*4+i]++;
        }
        for (int n=1; n<9; n++)
            for (int i=0; i<4; i++)
                local_moments[n*4 + i] += (pow_tmp[i] = val_tmp[i]*pow_tmp[i]);
        );
    MPI_Allreduce(
            (void*)local_moments,
            (void*)moments,
            4,
            MPI_DOUBLE, MPI_MIN, this->cd->comm);
    MPI_Allreduce(
            (void*)(local_moments + 4),
            (void*)(moments+4),
            8*4,
            MPI_DOUBLE, MPI_SUM, this->cd->comm);
    MPI_Allreduce(
            (void*)(local_moments + 9*4),
            (void*)(moments+9*4),
            4,
            MPI_DOUBLE, MPI_MAX, this->cd->comm);
    MPI_Allreduce(
            (void*)local_hist,
            (void*)hist,
            nbins*4,
            MPI_INT64_T, MPI_SUM, this->cd->comm);
    for (int n=1; n<9; n++)
        for (int i=0; i<4; i++)
            moments[n*4 + i] /= this->normalization_factor;
    fftw_free(local_moments);
    delete[] local_hist;
}

template <class rnumber>
void fluid_solver_base<rnumber>::write_spectrum(const char *fname, cnumber *a, const double k2exponent)
{
    double *spec = fftw_alloc_real(this->nshells);
    this->cospectrum(a, a, spec, k2exponent);
    if (this->cd->myrank == 0)
    {
        FILE *spec_file;
        char full_name[512];
        sprintf(full_name, "%s_%s_spec", this->name, fname);
        spec_file = fopen(full_name, "ab");
        fwrite((void*)&this->iteration, sizeof(int), 1, spec_file);
        fwrite((void*)spec, sizeof(double), this->nshells, spec_file);
        fclose(spec_file);
    }
    fftw_free(spec);
}

/*****************************************************************************/
/* macro for specializations to numeric types compatible with FFTW           */

#define FLUID_SOLVER_BASE_DEFINITIONS(FFTW, R, MPI_RNUM, MPI_CNUM) \
 \
template<> \
fluid_solver_base<R>::fluid_solver_base( \
        const char *NAME, \
        int nx, \
        int ny, \
        int nz, \
        double DKX, \
        double DKY, \
        double DKZ, \
        int DEALIAS_TYPE) \
{ \
    strncpy(this->name, NAME, 256); \
    this->name[255] = '\0'; \
    this->iteration = 0; \
 \
    int ntmp[4]; \
    ntmp[0] = nz; \
    ntmp[1] = ny; \
    ntmp[2] = nx; \
    ntmp[3] = 3; \
    this->rd = new field_descriptor<R>( \
            4, ntmp, MPI_RNUM, MPI_COMM_WORLD);\
    this->normalization_factor = (this->rd->full_size/3); \
    ntmp[0] = ny; \
    ntmp[1] = nz; \
    ntmp[2] = nx/2 + 1; \
    ntmp[3] = 3; \
    this->cd = new field_descriptor<R>( \
            4, ntmp, MPI_CNUM, this->rd->comm);\
 \
    this->dkx = DKX; \
    this->dky = DKY; \
    this->dkz = DKZ; \
    this->kx = new double[this->cd->sizes[2]]; \
    this->ky = new double[this->cd->subsizes[0]]; \
    this->kz = new double[this->cd->sizes[1]]; \
    this->dealias_type = DEALIAS_TYPE; \
    switch(this->dealias_type) \
    { \
        /* HL07 smooth filter */ \
        case 1: \
            this->kMx = this->dkx*(int(this->rd->sizes[2] / 2)); \
            this->kMy = this->dky*(int(this->rd->sizes[1] / 2)); \
            this->kMz = this->dkz*(int(this->rd->sizes[0] / 2)); \
            break; \
        default: \
            this->kMx = this->dkx*(int(this->rd->sizes[2] / 3)-1); \
            this->kMy = this->dky*(int(this->rd->sizes[1] / 3)-1); \
            this->kMz = this->dkz*(int(this->rd->sizes[0] / 3)-1); \
    } \
    int i, ii; \
    for (i = 0; i<this->cd->sizes[2]; i++) \
        this->kx[i] = i*this->dkx; \
    for (i = 0; i<this->cd->subsizes[0]; i++) \
    { \
        ii = i + this->cd->starts[0]; \
        if (ii <= this->rd->sizes[1]/2) \
            this->ky[i] = this->dky*ii; \
        else \
            this->ky[i] = this->dky*(ii - this->rd->sizes[1]); \
    } \
    for (i = 0; i<this->cd->sizes[1]; i++) \
    { \
        if (i <= this->rd->sizes[0]/2) \
            this->kz[i] = this->dkz*i; \
        else \
            this->kz[i] = this->dkz*(i - this->rd->sizes[0]); \
    } \
    this->kM = this->kMx; \
    if (this->kM < this->kMy) this->kM = this->kMy; \
    if (this->kM < this->kMz) this->kM = this->kMz; \
    this->kM2 = this->kM * this->kM; \
    this->dk = this->dkx; \
    if (this->dk > this->dky) this->dk = this->dky; \
    if (this->dk > this->dkz) this->dk = this->dkz; \
    this->dk2 = this->dk*this->dk; \
    DEBUG_MSG( \
            "kM = %g, kM2 = %g, dk = %g, dk2 = %g\n", \
            this->kM, this->kM2, this->dk, this->dk2); \
    /* spectra stuff */ \
    this->nshells = int(this->kM / this->dk) + 2; \
    this->kshell = new double[this->nshells]; \
    std::fill_n(this->kshell, this->nshells, 0.0); \
    this->nshell = new int64_t[this->nshells]; \
    std::fill_n(this->nshell, this->nshells, 0); \
    double *kshell_local = new double[this->nshells]; \
    std::fill_n(kshell_local, this->nshells, 0.0); \
    int64_t *nshell_local = new int64_t[this->nshells]; \
    std::fill_n(nshell_local, this->nshells, 0.0); \
    double knorm; \
    CLOOP_K2_NXMODES( \
            if (k2 < this->kM2) \
            { \
                knorm = sqrt(k2); \
                nshell_local[int(knorm/this->dk)] += nxmodes; \
                kshell_local[int(knorm/this->dk)] += nxmodes*knorm; \
            } \
            this->Fourier_filter[int(round(k2 / this->dk2))] = exp(-36.0 * pow(k2/this->kM2, 18.)); \
            ); \
    \
    MPI_Allreduce( \
            (void*)(nshell_local), \
            (void*)(this->nshell), \
            this->nshells, \
            MPI_INT64_T, MPI_SUM, this->cd->comm); \
    MPI_Allreduce( \
            (void*)(kshell_local), \
            (void*)(this->kshell), \
            this->nshells, \
            MPI_DOUBLE, MPI_SUM, this->cd->comm); \
    for (int n=0; n<this->nshells; n++) \
    { \
        this->kshell[n] /= this->nshell[n]; \
    } \
    delete[] nshell_local; \
    delete[] kshell_local; \
} \
 \
template<> \
fluid_solver_base<R>::~fluid_solver_base() \
{ \
    DEBUG_MSG("entered ~fluid_solver_base\n"); \
    delete[] this->kshell; \
    delete[] this->nshell; \
 \
    delete[] this->kx; \
    delete[] this->ky; \
    delete[] this->kz; \
 \
    delete this->cd; \
    delete this->rd; \
    DEBUG_MSG("exiting ~fluid_solver_base\n"); \
} \
 \
template<> \
void fluid_solver_base<R>::low_pass_Fourier(FFTW(complex) *a, const int howmany, const double kmax) \
{ \
    const double km2 = kmax*kmax; \
    const int howmany2 = 2*howmany; \
    /*DEBUG_MSG("entered low_pass_Fourier, kmax=%lg km2=%lg howmany2=%d\n", kmax, km2, howmany2);*/ \
    CLOOP_K2( \
            /*DEBUG_MSG("kx=%lg ky=%lg kz=%lg k2=%lg\n", \
                      this->kx[xindex], \
                      this->ky[yindex], \
                      this->kz[zindex], \
                      k2);*/ \
            if (k2 >= km2) \
                std::fill_n((R*)(a + howmany*cindex), howmany2, 0.0); \
            );\
} \
 \
template<> \
void fluid_solver_base<R>::dealias(FFTW(complex) *a, const int howmany) \
{ \
    if (this->dealias_type == 0) \
        { \
            this->low_pass_Fourier(a, howmany, this->kM); \
            return; \
        } \
    double tval; \
    CLOOP_K2( \
            tval = this->Fourier_filter[int(round(k2/this->dk2))]; \
            for (int tcounter = 0; tcounter < howmany; tcounter++) \
            for (int i=0; i<2; i++) \
                a[howmany*cindex+tcounter][i] *= tval; \
         ); \
} \
 \
template<> \
void fluid_solver_base<R>::force_divfree(FFTW(complex) *a) \
{ \
    FFTW(complex) tval; \
    CLOOP_K2( \
            { \
                tval[0] = (this->kx[xindex]*((*(a + cindex*3  ))[0]) + \
                           this->ky[yindex]*((*(a + cindex*3+1))[0]) + \
                           this->kz[zindex]*((*(a + cindex*3+2))[0]) ) / k2; \
                tval[1] = (this->kx[xindex]*((*(a + cindex*3  ))[1]) + \
                           this->ky[yindex]*((*(a + cindex*3+1))[1]) + \
                           this->kz[zindex]*((*(a + cindex*3+2))[1]) ) / k2; \
                for (int imag_part=0; imag_part<2; imag_part++) \
                { \
                    a[cindex*3  ][imag_part] -= tval[imag_part]*this->kx[xindex]; \
                    a[cindex*3+1][imag_part] -= tval[imag_part]*this->ky[yindex]; \
                    a[cindex*3+2][imag_part] -= tval[imag_part]*this->kz[zindex]; \
                } \
            } \
            );\
    if (this->cd->myrank == this->cd->rank[0]) \
        std::fill_n((R*)(a), 6, 0.0); \
} \
 \
template<> \
void fluid_solver_base<R>::symmetrize(FFTW(complex) *data, const int howmany) \
{ \
    ptrdiff_t ii, cc; \
    MPI_Status *mpistatus = new MPI_Status; \
    if (this->cd->myrank == this->cd->rank[0]) \
    { \
        for (cc = 0; cc < howmany; cc++) \
            data[cc][1] = 0.0; \
        for (ii = 1; ii < this->cd->sizes[1]/2; ii++) \
            for (cc = 0; cc < howmany; cc++) { \
                ( *(data + cc + howmany*(this->cd->sizes[1] - ii)*this->cd->sizes[2]))[0] = \
                 (*(data + cc + howmany*(                     ii)*this->cd->sizes[2]))[0]; \
                ( *(data + cc + howmany*(this->cd->sizes[1] - ii)*this->cd->sizes[2]))[1] = \
                -(*(data + cc + howmany*(                     ii)*this->cd->sizes[2]))[1]; \
                } \
    } \
    FFTW(complex) *buffer; \
    buffer = FFTW(alloc_complex)(howmany*this->cd->sizes[1]); \
    ptrdiff_t yy; \
    /*ptrdiff_t tindex;*/ \
    int ranksrc, rankdst; \
    for (yy = 1; yy < this->cd->sizes[0]/2; yy++) { \
        ranksrc = this->cd->rank[yy]; \
        rankdst = this->cd->rank[this->cd->sizes[0] - yy]; \
        if (this->cd->myrank == ranksrc) \
            for (ii = 0; ii < this->cd->sizes[1]; ii++) \
                for (cc = 0; cc < howmany; cc++) \
                    for (int imag_comp=0; imag_comp<2; imag_comp++) \
                    (*(buffer + howmany*ii+cc))[imag_comp] = \
                        (*(data + howmany*((yy - this->cd->starts[0])*this->cd->sizes[1] + ii)*this->cd->sizes[2] + cc))[imag_comp]; \
        if (ranksrc != rankdst) \
        { \
            DEBUG_MSG("inside fluid_solver_base::symmetrize, about to send/recv data\n"); \
            if (this->cd->myrank == ranksrc) \
                MPI_Send((void*)buffer, \
                         howmany*this->cd->sizes[1], MPI_CNUM, rankdst, yy, \
                         this->cd->comm); \
            if (this->cd->myrank == rankdst) \
                MPI_Recv((void*)buffer, \
                         howmany*this->cd->sizes[1], MPI_CNUM, ranksrc, yy, \
                         this->cd->comm, mpistatus); \
            DEBUG_MSG("inside fluid_solver_base::symmetrize, after send/recv data\n"); \
        } \
        if (this->cd->myrank == rankdst) \
        { \
            for (ii = 1; ii < this->cd->sizes[1]; ii++) \
                for (cc = 0; cc < howmany; cc++) \
                { \
                    (*(data + howmany*((this->cd->sizes[0] - yy - this->cd->starts[0])*this->cd->sizes[1] + ii)*this->cd->sizes[2] + cc))[0] = \
                        (*(buffer + howmany*(this->cd->sizes[1]-ii)+cc))[0]; \
                    (*(data + howmany*((this->cd->sizes[0] - yy - this->cd->starts[0])*this->cd->sizes[1] + ii)*this->cd->sizes[2] + cc))[1] = \
                       -(*(buffer + howmany*(this->cd->sizes[1]-ii)+cc))[1]; \
                } \
            for (cc = 0; cc < howmany; cc++) \
            { \
                (*((data + cc + howmany*(this->cd->sizes[0] - yy - this->cd->starts[0])*this->cd->sizes[1]*this->cd->sizes[2])))[0] =  (*(buffer + cc))[0]; \
                (*((data + cc + howmany*(this->cd->sizes[0] - yy - this->cd->starts[0])*this->cd->sizes[1]*this->cd->sizes[2])))[1] = -(*(buffer + cc))[1]; \
            } \
        } \
    } \
    FFTW(free)(buffer); \
    delete mpistatus; \
    /* put asymmetric data to 0 */\
    /*if (this->cd->myrank == this->cd->rank[this->cd->sizes[0]/2]) \
    { \
        tindex = howmany*(this->cd->sizes[0]/2 - this->cd->starts[0])*this->cd->sizes[1]*this->cd->sizes[2]; \
        for (ii = 0; ii < this->cd->sizes[1]; ii++) \
        { \
            std::fill_n((R*)(data + tindex), howmany*2*this->cd->sizes[2], 0.0); \
            tindex += howmany*this->cd->sizes[2]; \
        } \
    } \
    tindex = howmany*(); \
    std::fill_n((R*)(data + tindex), howmany*2, 0.0);*/ \
} \
 \
template<> \
int fluid_solver_base<R>::read_base(const char *fname, R *data) \
{ \
    char full_name[512]; \
    sprintf(full_name, "%s_%s_i%.5x", this->name, fname, this->iteration); \
    return this->rd->read(full_name, (void*)data); \
} \
 \
template<> \
int fluid_solver_base<R>::read_base(const char *fname, FFTW(complex) *data) \
{ \
    char full_name[512]; \
    sprintf(full_name, "%s_%s_i%.5x", this->name, fname, this->iteration); \
    return this->cd->read(full_name, (void*)data); \
} \
 \
template<> \
int fluid_solver_base<R>::write_base(const char *fname, R *data) \
{ \
    char full_name[512]; \
    sprintf(full_name, "%s_%s_i%.5x", this->name, fname, this->iteration); \
    return this->rd->write(full_name, (void*)data); \
} \
 \
template<> \
int fluid_solver_base<R>::write_base(const char *fname, FFTW(complex) *data) \
{ \
    char full_name[512]; \
    sprintf(full_name, "%s_%s_i%.5x", this->name, fname, this->iteration); \
    return this->cd->write(full_name, (void*)data); \
}

/*****************************************************************************/



/*****************************************************************************/
/* now actually use the macro defined above                                  */
FLUID_SOLVER_BASE_DEFINITIONS(
        FFTW_MANGLE_FLOAT,
        float,
        MPI_FLOAT,
        MPI_COMPLEX)
FLUID_SOLVER_BASE_DEFINITIONS(
        FFTW_MANGLE_DOUBLE,
        double,
        MPI_DOUBLE,
        BFPS_MPICXX_DOUBLE_COMPLEX)
/*****************************************************************************/



/*****************************************************************************/
/* finally, force generation of code for single precision                    */
template class fluid_solver_base<float>;
template class fluid_solver_base<double>;
/*****************************************************************************/

