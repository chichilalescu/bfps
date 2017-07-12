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
#include "shared_array.hpp"

template <field_backend be,
          kspace_dealias_type dt>
template <field_components fc>
kspace<be, dt>::kspace(
        const field_layout<fc> *source_layout,
        const double DKX,
        const double DKY,
        const double DKZ)
{
    TIMEZONE("kspace::kspace");
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

    shared_array<double> kshell_local_thread(this->nshells,[&](double* kshell_local){
        std::fill_n(kshell_local, this->nshells, 0);
    });
    shared_array<int64_t> nshell_local_thread(this->nshells,[&](int64_t* nshell_local){
        std::fill_n(nshell_local, this->nshells, 0);
    });

    std::vector<std::unordered_map<int, double>> dealias_filter_threaded(omp_get_max_threads());

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
                kshell_local_thread.getMine()[int(knorm/this->dk)] += nxmodes*knorm;
                nshell_local_thread.getMine()[int(knorm/this->dk)] += nxmodes;
            }
            if (dt == SMOOTH){
                dealias_filter_threaded[omp_get_thread_num()][int(round(k2 / this->dk2))] = exp(-36.0 * pow(k2/this->kM2, 18.));
            }
    });

    // Merge results

    kshell_local_thread.mergeParallel();
    nshell_local_thread.mergeParallel();

    if (dt == SMOOTH){
        for(int idxMerge = 0 ; idxMerge < int(dealias_filter_threaded.size()) ; ++idxMerge){
            for(const auto kv : dealias_filter_threaded[idxMerge]){
                this->dealias_filter[kv.first] = kv.second;
            }
        }
    }

    MPI_Allreduce(
            nshell_local_thread.getMasterData(),
            &this->nshell.front(),
            this->nshells,
            MPI_INT64_T, MPI_SUM, this->layout->comm);
    MPI_Allreduce(
            kshell_local_thread.getMasterData(),
            &this->kshell.front(),
            this->nshells,
            MPI_DOUBLE, MPI_SUM, this->layout->comm);
    for (int n=0; n<this->nshells; n++){
		if(this->nshell[n] != 0){
	        this->kshell[n] /= this->nshell[n];
		}
    }
}

template <field_backend be,
          kspace_dealias_type dt>
kspace<be, dt>::~kspace()
{
    delete this->layout;
}

template <field_backend be,
          kspace_dealias_type dt>
int kspace<be, dt>::store(hid_t stat_file)
{
    TIMEZONE("kspace::store");
    assert(this->layout->myrank == 0);
    hsize_t dims[4];
    hid_t space, dset;
    // store kspace information
    dset = H5Dopen(stat_file, "/kspace/kshell", H5P_DEFAULT);
    space = H5Dget_space(dset);
    H5Sget_simple_extent_dims(space, dims, NULL);
    H5Sclose(space);
    if (this->nshells != int(dims[0]))
    {
        DEBUG_MSG(
                "ERROR: computed nshells %d not equal to data file nshells %d\n",
                this->nshells, dims[0]);
    }
    H5Dwrite(
            dset,
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            &this->kshell.front());
    H5Dclose(dset);
    dset = H5Dopen(
            stat_file,
            "/kspace/nshell",
            H5P_DEFAULT);
    H5Dwrite(
            dset,
            H5T_NATIVE_INT64,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            &this->nshell.front());
    H5Dclose(dset);
    dset = H5Dopen(stat_file, "/kspace/kM", H5P_DEFAULT);
    H5Dwrite(
            dset,
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            &this->kM);
    H5Dclose(dset);
    dset = H5Dopen(stat_file, "/kspace/dk", H5P_DEFAULT);
    H5Dwrite(dset,
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            &this->dk);
    H5Dclose(dset);
    return EXIT_SUCCESS;
}

template <field_backend be,
          kspace_dealias_type dt>
template <typename rnumber,
          field_components fc>
void kspace<be, dt>::low_pass(
        typename fftw_interface<rnumber>::complex *__restrict__ a,
        const double kmax)
{
    const double km2 = kmax*kmax;
    this->CLOOP_K2(
            [&](ptrdiff_t cindex,
                ptrdiff_t xindex,
                ptrdiff_t yindex,
                ptrdiff_t zindex,
                double k2){
            if (k2 >= km2)
                std::fill_n((rnumber*)(a + ncomp(fc)*cindex), 2*ncomp(fc), 0);
                });
}

/** \brief Filter a field using a ball shaped top hat filter.
 *
 *  Filter's mathematical expression in real space is as follows:
 *  \f[
 *       \phi^b_\ell(r) =
 *           \frac{1}{\ell^3}\frac{6}{\pi} H(\ell/2 - r)
 *  \f]
 *  with the corresponding Fourier space expression:
 *  \f[
 *       \hat{\phi^b_\ell}(k) =
 *       \frac{3}{2(k\ell/2)^3}
 *       \left(2\sin (k \ell/2) - k \ell \cos (k \ell/2)\right)
 *  \f]
 */
template <field_backend be,
          kspace_dealias_type dt>
template <typename rnumber,
          field_components fc>
void kspace<be, dt>::ball_filter(
        typename fftw_interface<rnumber>::complex *__restrict__ a,
        const double ell)
{
    const double prefactor0 = double(3) / pow(ell/2, 3);
    this->CLOOP_K2(
            [&](ptrdiff_t cindex,
                ptrdiff_t xindex,
                ptrdiff_t yindex,
                ptrdiff_t zindex,
                double k2){
                if (k2 > 0)
                {
                    double argument = sqrt(k2)*ell / 2;
                    double prefactor = prefactor0 / pow(k2, 1.5);
                    for (unsigned int tcounter=0; tcounter<2*ncomp(fc); tcounter++)
                        ((rnumber*)a)[2*ncomp(fc)*cindex + tcounter] *= (
                            prefactor *
                            (sin(argument) - argument * cos(argument)));
                }
                });
}

/** \brief Filter a field using a Gaussian kernel.
 *
 *  Filter's mathematical expression in Fourier space is as follows:
 *  \f[
 *      \hat{g}_\ell(\mathbf{k}) = \exp(-k^2 \sigma^2 / 2)
 *  \f]
 */
template <field_backend be,
          kspace_dealias_type dt>
template <typename rnumber,
          field_components fc>
void kspace<be, dt>::Gauss_filter(
        typename fftw_interface<rnumber>::complex *__restrict__ a,
        const double sigma)
{
    const double prefactor = - sigma*sigma/2;
    this->CLOOP_K2(
            [&](ptrdiff_t cindex,
                ptrdiff_t xindex,
                ptrdiff_t yindex,
                ptrdiff_t zindex,
                double k2){
                {
                    for (unsigned int tcounter=0; tcounter<2*ncomp(fc); tcounter++)
                        ((rnumber*)a)[2*ncomp(fc)*cindex + tcounter] *= exp(prefactor*k2);
                }
                });
}

/** \brief Filter a field.
 *
 *  This is a wrapper that can choose between a sharp Fourier spherical filter,
 *  a Gaussian filter and a sharp real space spherical filter.
 *
 *  Filter expressions in real space are as follows:
 *  \f{eqnarray*}{
 *       \phi^b_\ell(r) &=&
 *           \frac{1}{\ell^3}\frac{6}{\pi} H(\ell/2 - r) \\
 *       \phi^g_\ell(r) &=&
 *           \frac{1}{\sigma_\ell^3}\frac{1}{(2\pi)^{3/2}}
 *           \exp\left(-\frac{1}{2}\left(\frac{r}{\sigma_\ell}\right)^2\right) \\
 *       \phi^s_\ell(r) &=&
 *           \frac{1}{2 \pi^2 r^3}
 *           \left(\sin k_\ell r - k_\ell r \cos k_\ell r\right)
 *  \f}
 *  and the corresponding expressions in Fourier space are:
 *  \f{eqnarray*}{
 *       \hat{\phi^b_\ell}(k) &=&
 *       \frac{3}{2(k\ell/2)^3}
 *       \left(2\sin (k \ell/2) - k \ell \cos (k \ell/2)\right) \\
 *       \hat{\phi^g_\ell}(k) &=&
 *       \exp\left(-\frac{1}{2}k^2 \sigma_\ell^2\right) \\
 *       \hat{\phi^s_\ell}(k) &=& H(k_\ell - k)
 *  \f}
 *
 *  \f$ k_\ell \f$ is given as a parameter, and then we use
 *  \f[
 *      \ell = \pi / k_\ell,
 *      \sigma_\ell = \pi / k_\ell
 *  \f]
 *
 *  For the Gaussian filter this is the same convention used in
 *  \cite Buzzicotti2017 .
 *
 *  See also `filter_calibrated_ell`.
 */
template <field_backend be,
          kspace_dealias_type dt>
template <typename rnumber,
          field_components fc>
int kspace<be, dt>::filter(
        typename fftw_interface<rnumber>::complex *__restrict__ a,
        const double wavenumber,
        std::string filter_type)
{
    if (filter_type == std::string("sharp_Fourier_sphere"))
    {
        this->template low_pass<rnumber, fc>(
                a,
                wavenumber);
    }
    else if (filter_type == std::string("Gauss"))
    {
        this->template Gauss_filter<rnumber, fc>(
                a,
                2*acos(0.)/wavenumber);
    }
    else if (filter_type == std::string("ball"))
    {
        this->template ball_filter<rnumber, fc>(
                a,
                2*acos(0.)/wavenumber);
    }
    return EXIT_SUCCESS;
}

/** \brief Filter a field.
 *
 *  This is a wrapper that can choose between a sharp Fourier spherical filter,
 *  a Gaussian filter and a sharp real space spherical filter.
 *
 *  Filter expressions in real space are as follows:
 *  \f{eqnarray*}{
 *      \phi^b_\ell(r) &=&
 *          \frac{1}{\ell^3}\frac{6}{\pi} H(\ell/2 - r) \\
 *      \phi^g_\ell(r) &=&
 *          \frac{1}{\sigma_\ell^3}\frac{1}{(2\pi)^{3/2}}
 *          \exp\left(-\frac{1}{2}\left(\frac{r}{\sigma_\ell}\right)^2\right) \\
 *      \phi^s_\ell(r) &=&
 *          \frac{1}{2 \pi^2 r^3}
 *          \left(\sin k_\ell r - k_\ell r \cos k_\ell r\right)
 *  \f}
 *  and the corresponding expressions in Fourier space are:
 *  \f{eqnarray*}{
 *      \hat{\phi^b_\ell}(k) &=&
 *      \frac{3}{2(k\ell/2)^3}
 *      \left(2\sin (k \ell/2) - k \ell \cos (k \ell/2)\right) \\
 *      \hat{\phi^g_\ell}(k) &=&
 *      \exp\left(-\frac{1}{2}k^2 \sigma_\ell^2\right) \\
 *      \hat{\phi^s_\ell}(k) &=& H(k_\ell - k)
 *  \f}
 *
 *  \f$\sigma_\ell\f$ and \f$k_\ell\f$ are calibrated such that the energy of
 *  the fluctuations is approximately the same (within the inertial range)
 *  independently of the shape of the filter.
 *
 *  This was done by hand, see [INSERT CITATION HERE] for details, with the
 *  results:
 *
 *  \f[
 *      \sigma_\ell = 0.257 \ell,
 *      k_\ell = 5.5 / \ell
 *  \f]
 *
 */
template <field_backend be,
          kspace_dealias_type dt>
template <typename rnumber,
          field_components fc>
int kspace<be, dt>::filter_calibrated_ell(
        typename fftw_interface<rnumber>::complex *__restrict__ a,
        const double ell,
        std::string filter_type)
{
    if (filter_type == std::string("sharp_Fourier_sphere"))
    {
        this->template low_pass<rnumber, fc>(
                a,
                2.8 / ell);
    }
    else if (filter_type == std::string("Gauss"))
    {
        this->template Gauss_filter<rnumber, fc>(
                a,
                0.23*ell);
    }
    else if (filter_type == std::string("ball"))
    {
        this->template ball_filter<rnumber, fc>(
                a,
                ell);
    }
    return EXIT_SUCCESS;
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
            this->low_pass<rnumber, fc>(a, this->kM);
            break;
        case SMOOTH:
            this->CLOOP_K2(
                [&](ptrdiff_t cindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex,
                    double k2){
                    double tval = this->dealias_filter[int(round(k2 / this->dk2))];
                    for (unsigned int tcounter=0; tcounter<2*ncomp(fc); tcounter++)
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
    this->CLOOP_K2(
                [&](ptrdiff_t cindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex,
                    double k2){
                if (k2 > 0)
        {
                    typename fftw_interface<rnumber>::complex tval;
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
    shared_array<double> spec_local_thread(this->nshells*ncomp(fc)*ncomp(fc),[&](double* spec_local){
        std::fill_n(spec_local, this->nshells*ncomp(fc)*ncomp(fc), 0);
    });

    this->CLOOP_K2_NXMODES(
            [&](ptrdiff_t cindex,
                ptrdiff_t xindex,
                ptrdiff_t yindex,
                ptrdiff_t zindex,
                double k2,
                int nxmodes){
            if (k2 <= this->kM2)
            {
                double* spec_local = spec_local_thread.getMine();
                int tmp_int = int(sqrt(k2) / this->dk)*ncomp(fc)*ncomp(fc);
                for (hsize_t i=0; i<ncomp(fc); i++)
                for (hsize_t j=0; j<ncomp(fc); j++){
                    spec_local[tmp_int + i*ncomp(fc)+j] += nxmodes * (
                        (a[ncomp(fc)*cindex + i][0] * b[ncomp(fc)*cindex + j][0]) +
                        (a[ncomp(fc)*cindex + i][1] * b[ncomp(fc)*cindex + j][1]));
                }
            }
            });

    spec_local_thread.mergeParallel();

    std::vector<double> spec;
    spec.resize(this->nshells*ncomp(fc)*ncomp(fc), 0);
    MPI_Allreduce(
            spec_local_thread.getMasterData(),
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
        typename fftw_interface<float>::complex *__restrict__ a,
        const double kmax);
template void kspace<FFTW, SMOOTH>::low_pass<float, THREE>(
        typename fftw_interface<float>::complex *__restrict__ a,
        const double kmax);
template void kspace<FFTW, SMOOTH>::low_pass<float, THREExTHREE>(
        typename fftw_interface<float>::complex *__restrict__ a,
        const double kmax);

template void kspace<FFTW, SMOOTH>::low_pass<double, ONE>(
        typename fftw_interface<double>::complex *__restrict__ a,
        const double kmax);
template void kspace<FFTW, SMOOTH>::low_pass<double, THREE>(
        typename fftw_interface<double>::complex *__restrict__ a,
        const double kmax);
template void kspace<FFTW, SMOOTH>::low_pass<double, THREExTHREE>(
        typename fftw_interface<double>::complex *__restrict__ a,
        const double kmax);

template void kspace<FFTW, SMOOTH>::Gauss_filter<float, ONE>(
        typename fftw_interface<float>::complex *__restrict__ a,
        const double kmax);
template void kspace<FFTW, SMOOTH>::Gauss_filter<float, THREE>(
        typename fftw_interface<float>::complex *__restrict__ a,
        const double kmax);
template void kspace<FFTW, SMOOTH>::Gauss_filter<float, THREExTHREE>(
        typename fftw_interface<float>::complex *__restrict__ a,
        const double kmax);

template void kspace<FFTW, SMOOTH>::Gauss_filter<double, ONE>(
        typename fftw_interface<double>::complex *__restrict__ a,
        const double kmax);
template void kspace<FFTW, SMOOTH>::Gauss_filter<double, THREE>(
        typename fftw_interface<double>::complex *__restrict__ a,
        const double kmax);
template void kspace<FFTW, SMOOTH>::Gauss_filter<double, THREExTHREE>(
        typename fftw_interface<double>::complex *__restrict__ a,
        const double kmax);

template int kspace<FFTW, SMOOTH>::filter<float, ONE>(
        typename fftw_interface<float>::complex *__restrict__ a,
        const double kmax,
        std::string filter_type);
template int kspace<FFTW, SMOOTH>::filter<float, THREE>(
        typename fftw_interface<float>::complex *__restrict__ a,
        const double kmax,
        std::string filter_type);
template int kspace<FFTW, SMOOTH>::filter<float, THREExTHREE>(
        typename fftw_interface<float>::complex *__restrict__ a,
        const double kmax,
        std::string filter_type);

template int kspace<FFTW, SMOOTH>::filter<double, ONE>(
        typename fftw_interface<double>::complex *__restrict__ a,
        const double kmax,
        std::string filter_type);
template int kspace<FFTW, SMOOTH>::filter<double, THREE>(
        typename fftw_interface<double>::complex *__restrict__ a,
        const double kmax,
        std::string filter_type);
template int kspace<FFTW, SMOOTH>::filter<double, THREExTHREE>(
        typename fftw_interface<double>::complex *__restrict__ a,
        const double kmax,
        std::string filter_type);

template int kspace<FFTW, SMOOTH>::filter_calibrated_ell<float, ONE>(
        typename fftw_interface<float>::complex *__restrict__ a,
        const double kmax,
        std::string filter_type);
template int kspace<FFTW, SMOOTH>::filter_calibrated_ell<float, THREE>(
        typename fftw_interface<float>::complex *__restrict__ a,
        const double kmax,
        std::string filter_type);
template int kspace<FFTW, SMOOTH>::filter_calibrated_ell<float, THREExTHREE>(
        typename fftw_interface<float>::complex *__restrict__ a,
        const double kmax,
        std::string filter_type);

template int kspace<FFTW, SMOOTH>::filter_calibrated_ell<double, ONE>(
        typename fftw_interface<double>::complex *__restrict__ a,
        const double kmax,
        std::string filter_type);
template int kspace<FFTW, SMOOTH>::filter_calibrated_ell<double, THREE>(
        typename fftw_interface<double>::complex *__restrict__ a,
        const double kmax,
        std::string filter_type);
template int kspace<FFTW, SMOOTH>::filter_calibrated_ell<double, THREExTHREE>(
        typename fftw_interface<double>::complex *__restrict__ a,
        const double kmax,
        std::string filter_type);

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

template void kspace<FFTW, TWO_THIRDS>::cospectrum<float, ONE>(
        const typename fftw_interface<float>::complex *__restrict__ a,
        const typename fftw_interface<float>::complex *__restrict__ b,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset);
template void kspace<FFTW, TWO_THIRDS>::cospectrum<float, THREE>(
        const typename fftw_interface<float>::complex *__restrict__ a,
        const typename fftw_interface<float>::complex *__restrict__ b,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset);
template void kspace<FFTW, TWO_THIRDS>::cospectrum<float, THREExTHREE>(
        const typename fftw_interface<float>::complex *__restrict__ a,
        const typename fftw_interface<float>::complex *__restrict__ b,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset);
template void kspace<FFTW, TWO_THIRDS>::cospectrum<double, ONE>(
        const typename fftw_interface<double>::complex *__restrict__ a,
        const typename fftw_interface<double>::complex *__restrict__ b,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset);
template void kspace<FFTW, TWO_THIRDS>::cospectrum<double, THREE>(
        const typename fftw_interface<double>::complex *__restrict__ a,
        const typename fftw_interface<double>::complex *__restrict__ b,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset);
template void kspace<FFTW, TWO_THIRDS>::cospectrum<double, THREExTHREE>(
        const typename fftw_interface<double>::complex *__restrict__ a,
        const typename fftw_interface<double>::complex *__restrict__ b,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset);

template void kspace<FFTW, SMOOTH>::cospectrum<float, ONE>(
        const typename fftw_interface<float>::complex *__restrict__ a,
        const typename fftw_interface<float>::complex *__restrict__ b,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset);
template void kspace<FFTW, SMOOTH>::cospectrum<float, THREE>(
        const typename fftw_interface<float>::complex *__restrict__ a,
        const typename fftw_interface<float>::complex *__restrict__ b,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset);
template void kspace<FFTW, SMOOTH>::cospectrum<float, THREExTHREE>(
        const typename fftw_interface<float>::complex *__restrict__ a,
        const typename fftw_interface<float>::complex *__restrict__ b,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset);
template void kspace<FFTW, SMOOTH>::cospectrum<double, ONE>(
        const typename fftw_interface<double>::complex *__restrict__ a,
        const typename fftw_interface<double>::complex *__restrict__ b,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset);
template void kspace<FFTW, SMOOTH>::cospectrum<double, THREE>(
        const typename fftw_interface<double>::complex *__restrict__ a,
        const typename fftw_interface<double>::complex *__restrict__ b,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset);
template void kspace<FFTW, SMOOTH>::cospectrum<double, THREExTHREE>(
        const typename fftw_interface<double>::complex *__restrict__ a,
        const typename fftw_interface<double>::complex *__restrict__ b,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset);

template void kspace<FFTW, SMOOTH>::force_divfree<float>(
       typename fftw_interface<float>::complex *__restrict__ a);
template void kspace<FFTW, SMOOTH>::force_divfree<double>(
       typename fftw_interface<double>::complex *__restrict__ a);

