/***********************************************************************
*
*  Copyright 2015 Max Planck Institute for Dynamics and SelfOrganization
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Contact: Cristian.Lalescu@ds.mpg.de
*
************************************************************************/



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
    sprintf(destination, "%s_%s_i%.5x", this->name, base_name, this->iteration); \
}

template <class rnumber>
void fluid_solver_base<rnumber>::clean_up_real_space(rnumber *a, int howmany)
{
    for (ptrdiff_t rindex = 0; rindex < this->cd->local_size*2; rindex += howmany*(this->rd->subsizes[2]+2))
        std::fill_n(a+rindex+this->rd->subsizes[2]*howmany, 2*howmany, 0.0);
}

template <class rnumber>
void fluid_solver_base<rnumber>::cospectrum(cnumber *a, cnumber *b, double *spec, const double k2exponent)
{
    double *cospec_local = fftw_alloc_real(this->nshells);
    std::fill_n(cospec_local, this->nshells, 0);
    double k2, knorm;
    int factor = 1;
    CLOOP(
            k2 = (this->kx[xindex]*this->kx[xindex] +
                  this->ky[yindex]*this->ky[yindex] +
                  this->kz[zindex]*this->kz[zindex]);
            if (k2 < this->kM2)
            {
                factor = (xindex == 0) ? 1 : 2;
                knorm = sqrt(k2);
                cospec_local[int(knorm/this->dk)] += factor * pow(k2, k2exponent) * (
                        (*(a + 3*cindex  ))[0] * (*(b + 3*cindex  ))[0] +
                        (*(a + 3*cindex  ))[1] * (*(b + 3*cindex  ))[1] +
                        (*(a + 3*cindex+1))[0] * (*(b + 3*cindex+1))[0] +
                        (*(a + 3*cindex+1))[1] * (*(b + 3*cindex+1))[1] +
                        (*(a + 3*cindex+2))[0] * (*(b + 3*cindex+2))[0] +
                        (*(a + 3*cindex+2))[1] * (*(b + 3*cindex+2))[1]
                                        );
            }
            );
    MPI_Allreduce(
            (void*)cospec_local,
            (void*)spec,
            this->nshells,
            MPI_DOUBLE, MPI_SUM, this->cd->comm);
    for (int n=0; n<this->nshells; n++)
    {
        spec[n] *= 12.5663706144*pow(this->kshell[n], 2) / this->nshell[n];
        /*is normalization needed?
         * spec[n] /= this->normalization_factor*/
    }
    fftw_free(cospec_local);
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

#define FLUID_SOLVER_BASE_DEFINITIONS(FFTW, R, C, MPI_RNUM, MPI_CNUM) \
 \
template<> \
fluid_solver_base<R>::fluid_solver_base( \
        const char *NAME, \
        int nx, \
        int ny, \
        int nz, \
        double DKX, \
        double DKY, \
        double DKZ) \
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
    this->knullx = new bool[this->cd->sizes[2]]; \
    this->knully = new bool[this->cd->subsizes[0]]; \
    this->knullz = new bool[this->cd->sizes[1]]; \
    this->nonzerokx = int(this->rd->sizes[2] / 3); \
    this->kMx = this->dkx*(this->nonzerokx-1); \
    this->nonzeroky = int(this->rd->sizes[1] / 3); \
    this->kMy = this->dky*(this->nonzeroky-1); \
    this->nonzeroky = 2*this->nonzeroky - 1; \
    this->nonzerokz = int(this->rd->sizes[0] / 3); \
    this->kMz = this->dkz*(this->nonzerokz-1); \
    this->nonzerokz = 2*this->nonzerokz - 1; \
    int i, ii; \
    for (i = 0; i<this->cd->sizes[2]; i++) \
    { \
        this->kx[i] = i*this->dkx; \
        if (i < this->nonzerokx) \
            this->knullx[i] = false; \
        else \
            this->knullx[i] = true; \
    } \
    for (i = 0; i<this->cd->subsizes[0]; i++) \
    { \
        int tval = (this->nonzeroky+1)/2; \
        ii = i + this->cd->starts[0]; \
        if (ii <= this->rd->sizes[1]/2) \
            this->ky[i] = this->dky*ii; \
        else \
            this->ky[i] = this->dky*(ii - this->rd->sizes[1]); \
        if (ii < tval || (this->rd->sizes[1] - ii) < tval) \
            this->knully[i] = false; \
        else \
            this->knully[i] = true; \
    } \
    for (i = 0; i<this->cd->sizes[1]; i++) \
    { \
        int tval = (this->nonzerokz+1)/2; \
        if (i <= this->rd->sizes[0]/2) \
            this->kz[i] = this->dkz*i; \
        else \
            this->kz[i] = this->dkz*(i - this->rd->sizes[0]); \
        if (i < tval || (this->rd->sizes[0] - i) < tval) \
            this->knullz[i] = false; \
        else \
            this->knullz[i] = true; \
    } \
    this->kM = this->kMx; \
    if (this->kM < this->kMy) this->kM = this->kMy; \
    if (this->kM < this->kMz) this->kM = this->kMz; \
    this->kM2 = this->kM * this->kM; \
    this->dk = this->dkx; \
    if (this->dk > this->dky) this->dk = this->dky; \
    if (this->dk > this->dkz) this->dk = this->dkz; \
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
    double k2; \
    int nxmodes; \
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            if (k2 < this->kM2) \
            { \
                k2 = sqrt(k2); \
                nxmodes = (xindex == 0) ? 1 : 2; \
                nshell_local[int(k2/this->dk)] += nxmodes; \
                kshell_local[int(k2/this->dk)] += nxmodes*k2; \
            } \
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
    /* output kshells for this simulation */ \
    if (this->cd->myrank == 0) \
    { \
        char fname[512]; \
        sprintf(fname, "%s_kshell", this->name); \
        FILE *kshell_file; \
        kshell_file = fopen(fname, "wb"); \
        fwrite((void*)this->kshell, sizeof(double), this->nshells, kshell_file); \
        fclose(kshell_file); \
    } \
} \
 \
template<> \
fluid_solver_base<R>::~fluid_solver_base() \
{ \
    delete[] this->kshell; \
    delete[] this->nshell; \
 \
    delete[] this->kx;\
    delete[] this->ky;\
    delete[] this->kz;\
    delete[] this->knullx;\
    delete[] this->knully;\
    delete[] this->knullz;\
 \
    delete this->cd; \
    delete this->rd; \
} \
 \
template<> \
R fluid_solver_base<R>::correl_vec(C *a, C *b) \
{ \
    double val_process = 0.0, val; \
    double k2; \
    int factor = 1;\
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            if (k2 < this->kM2) \
            { \
                factor = (xindex == 0) ? 1 : 2; \
                val_process += factor * ((*(a + 3*cindex))[0] * (*(b + 3*cindex))[0] + \
                                         (*(a + 3*cindex))[1] * (*(b + 3*cindex))[1] + \
                                         (*(a + 3*cindex+1))[0] * (*(b + 3*cindex+1))[0] + \
                                         (*(a + 3*cindex+1))[1] * (*(b + 3*cindex+1))[1] + \
                                         (*(a + 3*cindex+2))[0] * (*(b + 3*cindex+2))[0] + \
                                         (*(a + 3*cindex+2))[1] * (*(b + 3*cindex+2))[1] \
                                        ); \
            } \
            );\
    MPI_Allreduce( \
            (void*)(&val_process), \
            (void*)(&val), \
            1, MPI_DOUBLE_PRECISION, MPI_SUM, this->cd->comm); \
    return R(val); \
} \
 \
template<> \
void fluid_solver_base<R>::low_pass_Fourier(C *a, const int howmany, const double kmax) \
{ \
    double k2; \
    const double km2 = kmax*kmax; \
    const int howmany2 = 2*howmany; \
    /*DEBUG_MSG("entered low_pass_Fourier, kmax=%lg km2=%lg howmany2=%d\n", kmax, km2, howmany2);*/ \
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            /*DEBUG_MSG("kx=%lg ky=%lg kz=%lg k2=%lg\n", \
                      this->kx[xindex], \
                      this->ky[yindex], \
                      this->kz[zindex], \
                      k2);*/ \
            if (k2 >= km2) \
            { \
                /*for (int tcounter = 0; tcounter < howmany; tcounter++) \
                { \
                    a[howmany*cindex+tcounter][0] = 0.0; \
                    a[howmany*cindex+tcounter][1] = 0.0; \
                }*/ \
                std::fill_n((R*)(a + howmany*cindex), howmany2, 0.0); \
            } \
            );\
} \
 \
template<> \
void fluid_solver_base<R>::force_divfree(C *a) \
{ \
    double k2; \
    C tval; \
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
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
void fluid_solver_base<R>::symmetrize(C *data, const int howmany) \
{ \
    ptrdiff_t ii, cc; \
    MPI_Status *mpistatus = new MPI_Status[1]; \
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
    C *buffer; \
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
            if (this->cd->myrank == ranksrc) \
                MPI_Send((void*)buffer, \
                         howmany*this->cd->sizes[1], MPI_CNUM, rankdst, yy, \
                         this->cd->comm); \
            if (this->cd->myrank == rankdst) \
                MPI_Recv((void*)buffer, \
                         howmany*this->cd->sizes[1], MPI_CNUM, ranksrc, yy, \
                         this->cd->comm, mpistatus); \
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
int fluid_solver_base<R>::read_base(const char *fname, C *data) \
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
int fluid_solver_base<R>::write_base(const char *fname, C *data) \
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
        fftwf_complex,
        MPI_FLOAT,
        MPI_COMPLEX)
//FLUID_SOLVER_BASE_DEFINITIONS(
//        FFTW_MANGLE_DOUBLE,
//        double,
//        fftw_complex,
//        MPI_DOUBLE,
//        MPI_C_DOUBLE_COMPLEX)
/*****************************************************************************/



/*****************************************************************************/
/* finally, force generation of code for single precision                    */
template class fluid_solver_base<float>;
/*****************************************************************************/

