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


#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include "kspace.hpp"
#include "scope_timer.hpp"



template <field_backend be,
          kspace_dealias_type dt>
template <field_components fc>
kspace<be, dt>::kspace(
        const field_layout<fc> *source_layout,
        const double DKX,
        const double DKY,
        const double DKZ)
{
    TIMEZONE("field::kspace");
    /* get layout */
    this->layout = new field_layout<ONE>(
            source_layout->sizes,
            source_layout->subsizes,
            source_layout->starts,
            source_layout->comm);

    /* store dk values */
    this->dkx = DKX;
    this->dky = DKY;
    this->dkz = DKZ;

    /* compute kx, ky, kz and compute kM values */
    switch(be)
    {
        case FFTW:
            this->kx.resize(this->layout->sizes[2]);
            this->ky.resize(this->layout->subsizes[0]);
            this->kz.resize(this->layout->sizes[1]);
            int i, ii;
            for (i = 0; i<int(this->layout->sizes[2]); i++)
                this->kx[i] = i*this->dkx;
            for (i = 0; i<int(this->layout->subsizes[0]); i++)
            {
                ii = i + this->layout->starts[0];
                if (ii <= int(this->layout->sizes[1]/2))
                    this->ky[i] = this->dky*ii;
                else
                    this->ky[i] = this->dky*(ii - int(this->layout->sizes[1]));
            }
            for (i = 0; i<int(this->layout->sizes[1]); i++)
            {
                if (i <= int(this->layout->sizes[0]/2))
                    this->kz[i] = this->dkz*i;
                else
                    this->kz[i] = this->dkz*(i - int(this->layout->sizes[0]));
            }
            switch(dt)
            {
                case TWO_THIRDS:
                    this->kMx = this->dkx*(int(2*(int(this->layout->sizes[2])-1)/3)-1);
                    this->kMy = this->dky*(int(this->layout->sizes[0] / 3)-1);
                    this->kMz = this->dkz*(int(this->layout->sizes[1] / 3)-1);
                    break;
                case SMOOTH:
                    this->kMx = this->dkx*(int(this->layout->sizes[2])-2);
                    this->kMy = this->dky*(int(this->layout->sizes[0] / 2)-1);
                    this->kMz = this->dkz*(int(this->layout->sizes[1] / 2)-1);
                    break;
            }
            break;
    }

    /* get global kM and dk */
    this->kM = this->kMx;
    if (this->kM < this->kMy) this->kM = this->kMy;
    if (this->kM < this->kMz) this->kM = this->kMz;
    this->kM2 = this->kM * this->kM;
    this->dk = this->dkx;
    if (this->dk > this->dky) this->dk = this->dky;
    if (this->dk > this->dkz) this->dk = this->dkz;
    this->dk2 = this->dk*this->dk;

    /* spectra stuff */
    this->nshells = int(this->kM / this->dk) + 2;
    this->kshell.resize(this->nshells, 0);
    this->nshell.resize(this->nshells, 0);
    std::vector<double> kshell_local;
    kshell_local.resize(this->nshells, 0);
    std::vector<int64_t> nshell_local;
    nshell_local.resize(this->nshells, 0);
    this->CLOOP_K2_NXMODES(
            [&](ptrdiff_t cindex,
                ptrdiff_t xindex,
                ptrdiff_t yindex,
                ptrdiff_t zindex,
                double k2,
                int nxmodes){
            if (k2 < this->kM2)
            {
                double knorm = sqrt(k2);
                nshell_local[int(knorm/this->dk)] += nxmodes;
                kshell_local[int(knorm/this->dk)] += nxmodes*knorm;
            }
            if (dt == SMOOTH)
                this->dealias_filter[int(round(k2 / this->dk2))] = \
                    exp(-36.0 * pow(k2/this->kM2, 18.));
                });
    MPI_Allreduce(
            &nshell_local.front(),
            &this->nshell.front(),
            this->nshells,
            MPI_INT64_T, MPI_SUM, this->layout->comm);
    MPI_Allreduce(
            &kshell_local.front(),
            &this->kshell.front(),
            this->nshells,
            MPI_DOUBLE, MPI_SUM, this->layout->comm);
    for (int n=0; n<this->nshells; n++)
        this->kshell[n] /= this->nshell[n];
}

template <field_backend be,
          kspace_dealias_type dt>
kspace<be, dt>::~kspace()
{
    delete this->layout;
}

template <field_backend be,
          kspace_dealias_type dt>
template <typename rnumber,
          field_components fc>
void kspace<be, dt>::low_pass(rnumber *__restrict__ a, const double kmax)
{
    const double km2 = kmax*kmax;
    this->CLOOP_K2(
            [&](ptrdiff_t cindex,
                ptrdiff_t xindex,
                ptrdiff_t yindex,
                ptrdiff_t zindex,
                double k2){
            if (k2 >= km2)
                std::fill_n(a + 2*ncomp(fc)*cindex, 2*ncomp(fc), 0);
                });
}

template <field_backend be,
          kspace_dealias_type dt>
template <typename rnumber,
          field_components fc>
void kspace<be, dt>::dealias(typename fftw_interface<rnumber>::complex *__restrict__ a)
{
    switch(dt)
    {
        case TWO_THIRDS:
            this->low_pass<rnumber, fc>((rnumber*)a, this->kM);
            break;
        case SMOOTH:
            this->CLOOP_K2(
                [&](ptrdiff_t cindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex,
                    double k2){
                    double tval = this->dealias_filter[int(round(k2 / this->dk2))];
                    for (int tcounter=0; tcounter<2*ncomp(fc); tcounter++)
                        ((rnumber*)a)[2*ncomp(fc)*cindex + tcounter] *= tval;
                        });
            break;
    }
}

template <field_backend be,
          kspace_dealias_type dt>
template <typename rnumber>
void kspace<be, dt>::force_divfree(typename fftw_interface<rnumber>::complex *__restrict__ a)
{
    TIMEZONE("kspace::force_divfree");
    typename fftw_interface<rnumber>::complex tval;
    this->CLOOP_K2(
                [&](ptrdiff_t cindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex,
                    double k2){
                if (k2 > 0)
        {
            tval[0] = (this->kx[xindex]*((*(a + cindex*3  ))[0]) +
                       this->ky[yindex]*((*(a + cindex*3+1))[0]) +
                       this->kz[zindex]*((*(a + cindex*3+2))[0]) ) / k2;
            tval[1] = (this->kx[xindex]*((*(a + cindex*3  ))[1]) +
                       this->ky[yindex]*((*(a + cindex*3+1))[1]) +
                       this->kz[zindex]*((*(a + cindex*3+2))[1]) ) / k2;
            for (int imag_part=0; imag_part<2; imag_part++)
            {
                a[cindex*3  ][imag_part] -= tval[imag_part]*this->kx[xindex];
                a[cindex*3+1][imag_part] -= tval[imag_part]*this->ky[yindex];
                a[cindex*3+2][imag_part] -= tval[imag_part]*this->kz[zindex];
            }
        }
                }
    );
    if (this->layout->myrank == this->layout->rank[0][0])
        std::fill_n((rnumber*)(a), 6, 0.0);
}

template <field_backend be,
          kspace_dealias_type dt>
template <typename rnumber,
          field_components fc>
void kspace<be, dt>::cospectrum(
        const rnumber(* __restrict a)[2],
        const rnumber(* __restrict b)[2],
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset)
{
    TIMEZONE("field::cospectrum");
    std::vector<double> spec, spec_local;
    spec.resize(this->nshells*ncomp(fc)*ncomp(fc), 0);
    spec_local.resize(this->nshells*ncomp(fc)*ncomp(fc), 0);
    this->CLOOP_K2_NXMODES(
            [&](ptrdiff_t cindex,
                ptrdiff_t xindex,
                ptrdiff_t yindex,
                ptrdiff_t zindex,
                double k2,
                int nxmodes){
            if (k2 <= this->kM2)
            {
                int tmp_int = int(sqrt(k2) / this->dk)*ncomp(fc)*ncomp(fc);
                for (hsize_t i=0; i<ncomp(fc); i++)
                for (hsize_t j=0; j<ncomp(fc); j++)
                    spec_local[tmp_int + i*ncomp(fc)+j] += nxmodes * (
                    (a[ncomp(fc)*cindex + i][0] * b[ncomp(fc)*cindex + j][0]) +
                    (a[ncomp(fc)*cindex + i][1] * b[ncomp(fc)*cindex + j][1]));
            }
            });
    MPI_Allreduce(
            &spec_local.front(),
            &spec.front(),
            spec.size(),
            MPI_DOUBLE, MPI_SUM, this->layout->comm);
    if (this->layout->myrank == 0)
    {
        hid_t dset, wspace, mspace;
        hsize_t count[(ndim(fc)-2)*2], offset[(ndim(fc)-2)*2], dims[(ndim(fc)-2)*2];
        dset = H5Dopen(group, ("spectra/" + dset_name).c_str(), H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        H5Sget_simple_extent_dims(wspace, dims, NULL);
        switch (fc)
        {
            case THREExTHREE:
                offset[4] = 0;
                offset[5] = 0;
                count[4] = ncomp(fc);
                count[5] = ncomp(fc);
            case THREE:
                offset[2] = 0;
                offset[3] = 0;
                count[2] = ncomp(fc);
                count[3] = ncomp(fc);
            default:
                offset[0] = toffset;
                offset[1] = 0;
                count[0] = 1;
                count[1] = this->nshells;
        }
        mspace = H5Screate_simple((ndim(fc)-2)*2, count, NULL);
        H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, wspace, H5P_DEFAULT, &spec.front());
        H5Sclose(wspace);
        H5Sclose(mspace);
        H5Dclose(dset);
    }
}


template class kspace<FFTW, TWO_THIRDS>;
template class kspace<FFTW, SMOOTH>;

template kspace<FFTW, TWO_THIRDS>::kspace<>(
        const field_layout<ONE> *,
        const double, const double, const double);
template kspace<FFTW, TWO_THIRDS>::kspace<>(
        const field_layout<THREE> *,
        const double, const double, const double);
template kspace<FFTW, TWO_THIRDS>::kspace<>(
        const field_layout<THREExTHREE> *,
        const double, const double, const double);

template kspace<FFTW, SMOOTH>::kspace<>(
        const field_layout<ONE> *,
        const double, const double, const double);
template kspace<FFTW, SMOOTH>::kspace<>(
        const field_layout<THREE> *,
        const double, const double, const double);
template kspace<FFTW, SMOOTH>::kspace<>(
        const field_layout<THREExTHREE> *,
        const double, const double, const double);

template void kspace<FFTW, SMOOTH>::low_pass<float, ONE>(
        float *__restrict__ a,
        const double kmax);
template void kspace<FFTW, SMOOTH>::low_pass<float, THREE>(
        float *__restrict__ a,
        const double kmax);
template void kspace<FFTW, SMOOTH>::low_pass<float, THREExTHREE>(
        float *__restrict__ a,
        const double kmax);

template void kspace<FFTW, SMOOTH>::low_pass<double, ONE>(
        double *__restrict__ a,
        const double kmax);
template void kspace<FFTW, SMOOTH>::low_pass<double, THREE>(
        double *__restrict__ a,
        const double kmax);
template void kspace<FFTW, SMOOTH>::low_pass<double, THREExTHREE>(
        double *__restrict__ a,
        const double kmax);

template void kspace<FFTW, SMOOTH>::dealias<float, ONE>(
        typename fftw_interface<float>::complex *__restrict__ a);
template void kspace<FFTW, SMOOTH>::dealias<float, THREE>(
        typename fftw_interface<float>::complex *__restrict__ a);
template void kspace<FFTW, SMOOTH>::dealias<float, THREExTHREE>(
        typename fftw_interface<float>::complex *__restrict__ a);

template void kspace<FFTW, SMOOTH>::dealias<double, ONE>(
        typename fftw_interface<double>::complex *__restrict__ a);
template void kspace<FFTW, SMOOTH>::dealias<double, THREE>(
        typename fftw_interface<double>::complex *__restrict__ a);
template void kspace<FFTW, SMOOTH>::dealias<double, THREExTHREE>(
        typename fftw_interface<double>::complex *__restrict__ a);

template void kspace<FFTW, SMOOTH>::force_divfree<float>(
       typename fftw_interface<float>::complex *__restrict__ a);
template void kspace<FFTW, SMOOTH>::force_divfree<double>(
       typename fftw_interface<double>::complex *__restrict__ a);

