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



#define NDEBUG

#include <cassert>
#include <cmath>
#include <cstring>
#include "base.hpp"
#include "fluid_solver_base.hpp"
#include "fftw_tools.hpp"
#include "scope_timer.hpp"
#include "shared_array.hpp"

template <class rnumber>
void fluid_solver_base<rnumber>::fill_up_filename(const char *base_name, char *destination)
{
    sprintf(destination, "%s_%s_i%.5x", this->name, base_name, this->iteration);
}

template <class rnumber>
void fluid_solver_base<rnumber>::clean_up_real_space(rnumber *a, int howmany)
{
    TIMEZONE("fluid_solver_base::clean_up_real_space");
    for (ptrdiff_t rindex = 0; rindex < this->cd->local_size*2; rindex += howmany*(this->rd->subsizes[2]+2))
        std::fill_n(a+rindex+this->rd->subsizes[2]*howmany, 2*howmany, 0.0);
}

template <class rnumber>
double fluid_solver_base<rnumber>::autocorrel(cnumber *a)
{
    double *spec = fftw_alloc_real(this->nshells*9);
    double sum_local;
    this->cospectrum(a, a, spec);
    sum_local = 0.0;
    for (unsigned int n = 0; n < this->nshells; n++)
    {
        sum_local += spec[n*9] + spec[n*9 + 4] + spec[n*9 + 8];
    }
    fftw_free(spec);
    return sum_local;
}

template <class rnumber>
void fluid_solver_base<rnumber>::cospectrum(cnumber *a, cnumber *b, double *spec)
{
    TIMEZONE("fluid_solver_base::cospectrum");
    shared_array<double> cospec_local_thread(this->nshells*9,[&](double* cospec_local){
        std::fill_n(cospec_local, this->nshells*9, 0);
    });

    CLOOP_K2_NXMODES(
                this,

                [&](ptrdiff_t cindex, ptrdiff_t /*xindex*/, ptrdiff_t /*yindex*/,
                ptrdiff_t /*zindex*/, double k2, int nxmodes){
        if (k2 <= this->kMspec2)
        {
            int tmp_int = int(sqrt(k2)/this->dk)*9;
            double* cospec_local = cospec_local_thread.getMine();
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                {
                    cospec_local[tmp_int+i*3+j] += nxmodes * (
                                (*(a + 3*cindex+i))[0] * (*(b + 3*cindex+j))[0] +
                            (*(a + 3*cindex+i))[1] * (*(b + 3*cindex+j))[1]);
                }
        }}
    );
    cospec_local_thread.mergeParallel();
    MPI_Allreduce(
                cospec_local_thread.getMasterData(),
                (void*)spec,
                this->nshells*9,
                MPI_DOUBLE, MPI_SUM, this->cd->comm);
}

template <class rnumber>
void fluid_solver_base<rnumber>::cospectrum(cnumber *a, cnumber *b, double *spec, const double k2exponent)
{
    TIMEZONE("fluid_solver_base::cospectrum2");
    shared_array<double> cospec_local_thread(this->nshells*9,[&](double* cospec_local){
        std::fill_n(cospec_local, this->nshells*9, 0);
    });

    CLOOP_K2_NXMODES(
                this,

                [&](ptrdiff_t cindex, ptrdiff_t /*xindex*/, ptrdiff_t /*yindex*/,
                ptrdiff_t /*zindex*/, double k2, int nxmodes){
        if (k2 <= this->kMspec2)
        {
            double factor = nxmodes*pow(k2, k2exponent);
            int tmp_int = int(sqrt(k2)/this->dk)*9;
            double* cospec_local = cospec_local_thread.getMine();
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                {
                    cospec_local[tmp_int+i*3+j] += factor * (
                                (*(a + 3*cindex+i))[0] * (*(b + 3*cindex+j))[0] +
                            (*(a + 3*cindex+i))[1] * (*(b + 3*cindex+j))[1]);
                }
        }}
    );
    cospec_local_thread.mergeParallel();
    MPI_Allreduce(
                cospec_local_thread.getMasterData(),
                (void*)spec,
                this->nshells*9,
                MPI_DOUBLE, MPI_SUM, this->cd->comm);
    //for (int n=0; n<this->nshells; n++)
    //{
    //    spec[n] *= 12.5663706144*pow(this->kshell[n], 2) / this->nshell[n];
    //    /*is normalization needed?
    //     * spec[n] /= this->normalization_factor*/
    //}
}

template <class rnumber>
void fluid_solver_base<rnumber>::compute_rspace_stats(
        const rnumber *a,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset,
        const std::vector<double> max_estimate)
{
    TIMEZONE("fluid_solver_base::compute_rspace_stats");
    const int nmoments = 10;
    int nvals, nbins;
    if (this->rd->myrank == 0)
    {
        hid_t dset, wspace;
        hsize_t dims[3];
        int ndims;
        dset = H5Dopen(group, ("moments/" + dset_name).c_str(), H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        ndims = H5Sget_simple_extent_dims(wspace, dims, NULL);
        assert(ndims == 3);
        assert(dims[1] == nmoments);
        nvals = dims[2];
        H5Sclose(wspace);
        H5Dclose(dset);
        dset = H5Dopen(group, ("histograms/" + dset_name).c_str(), H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        ndims = H5Sget_simple_extent_dims(wspace, dims, NULL);
        assert(ndims == 3);
        nbins = dims[1];
        assert(nvals == dims[2]);
        H5Sclose(wspace);
        H5Dclose(dset);
    }
    MPI_Bcast(&nvals, 1, MPI_INT, 0, this->rd->comm);
    MPI_Bcast(&nbins, 1, MPI_INT, 0, this->rd->comm);
    assert(nvals == max_estimate.size());
    shared_array<double> threaded_local_moments(nmoments*nvals, [&](double* local_moments){
        std::fill_n(local_moments, nmoments*nvals, 0);
        if (nvals == 4) local_moments[3] = max_estimate[3];
    });

    shared_array<double> threaded_val_tmp(nvals);

    shared_array<ptrdiff_t> threaded_local_hist(nbins*nvals, [&](ptrdiff_t* local_hist){
        std::fill_n(local_hist, nbins*nvals, 0);
    });

    // Not written by threads
    double *binsize = new double[nvals];
    for (int i=0; i<nvals; i++)
        binsize[i] = 2*max_estimate[i] / nbins;

    RLOOP(
                this,
                [&](ptrdiff_t rindex, ptrdiff_t /*xindex*/, ptrdiff_t /*yindex*/, ptrdiff_t /*zindex*/){
        double *val_tmp = threaded_val_tmp.getMine();
        ptrdiff_t* local_hist = threaded_local_hist.getMine();
        double *local_moments = threaded_local_moments.getMine();

        if (nvals == 4) val_tmp[3] = 0.0;
        for (int i=0; i<3; i++)
        {
            val_tmp[i] = a[rindex*3+i];
            if (nvals == 4) val_tmp[3] += val_tmp[i]*val_tmp[i];
        }
        if (nvals == 4)
        {
            val_tmp[3] = sqrt(val_tmp[3]);
            if (val_tmp[3] < local_moments[0*nvals+3])
                local_moments[0*nvals+3] = val_tmp[3];
            if (val_tmp[3] > local_moments[9*nvals+3])
                local_moments[9*nvals+3] = val_tmp[3];
            int bin = int(floor(val_tmp[3]*2/binsize[3]));
            if (bin >= 0 && bin < nbins)
                local_hist[bin*nvals+3]++;
        }
        for (int i=0; i<3; i++)
        {
            if (val_tmp[i] < local_moments[0*nvals+i])
                local_moments[0*nvals+i] = val_tmp[i];
            if (val_tmp[i] > local_moments[(nmoments-1)*nvals+i])
                local_moments[(nmoments-1)*nvals+i] = val_tmp[i];
            int bin = int(floor((val_tmp[i] + max_estimate[i]) / binsize[i]));
            if (bin >= 0 && bin < nbins)
                local_hist[bin*nvals+i]++;
        }
        for (int n=1; n < nmoments-1; n++){
            double pow_tmp = 1.;
            for (int i=0; i<nvals; i++){
                local_moments[n*nvals + i] += (pow_tmp = val_tmp[i]*pow_tmp);
            }
        }
    }
    );

    threaded_local_hist.mergeParallel();
    threaded_local_moments.mergeParallel([&](const int idx, const double& v1, const double& v2) -> double {
          if(nvals == int(4) && idx == 0*nvals+3){
              return std::min(v1, v2);  
          }
          if(nvals == int(4) && idx == 9*nvals+3){
              return std::max(v1, v2);  
          }
          if(idx < 3){
              return std::min(v1, v2);        
          }      
          if((nmoments-1)*nvals <= idx && idx < (nmoments-1)*nvals+3){
              return std::max(v1, v2);        
          }
          return v1 + v2;
      });


    double *moments = new double[nmoments*nvals];
    MPI_Allreduce(
                threaded_local_moments.getMasterData(),
                (void*)moments,
                nvals,
                MPI_DOUBLE, MPI_MIN, this->cd->comm);
    MPI_Allreduce(
                (threaded_local_moments.getMasterData() + nvals),
                (void*)(moments+nvals),
                (nmoments-2)*nvals,
                MPI_DOUBLE, MPI_SUM, this->cd->comm);
    MPI_Allreduce(
                (threaded_local_moments.getMasterData() + (nmoments-1)*nvals),
                (void*)(moments+(nmoments-1)*nvals),
                nvals,
                MPI_DOUBLE, MPI_MAX, this->cd->comm);
    ptrdiff_t *hist = new ptrdiff_t[nbins*nvals];
    MPI_Allreduce(
                threaded_local_hist.getMasterData(),
                (void*)hist,
                nbins*nvals,
                MPI_INT64_T, MPI_SUM, this->cd->comm);
    for (int n=1; n < nmoments-1; n++)
        for (int i=0; i<nvals; i++)
            moments[n*nvals + i] /= this->normalization_factor;
    delete[] binsize;
    if (this->rd->myrank == 0)
    {
        hid_t dset, wspace, mspace;
        hsize_t count[3], offset[3], dims[3];
        dset = H5Dopen(group, ("moments/" + dset_name).c_str(), H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        H5Sget_simple_extent_dims(wspace, dims, NULL);
        offset[0] = toffset;
        offset[1] = 0;
        offset[2] = 0;
        count[0] = 1;
        count[1] = nmoments;
        count[2] = nvals;
        mspace = H5Screate_simple(3, count, NULL);
        H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, wspace, H5P_DEFAULT, moments);
        H5Sclose(wspace);
        H5Sclose(mspace);
        H5Dclose(dset);
        dset = H5Dopen(group, ("histograms/" + dset_name).c_str(), H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        count[1] = nbins;
        mspace = H5Screate_simple(3, count, NULL);
        H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Dwrite(dset, H5T_NATIVE_INT64, mspace, wspace, H5P_DEFAULT, hist);
        H5Sclose(wspace);
        H5Sclose(mspace);
        H5Dclose(dset);
    }
    delete[] moments;
    delete[] hist;
}



template <class rnumber>
template<int nvals>
void fluid_solver_base<rnumber>::compute_rspace_stats(
        rnumber *a,
        double *moments,
        ptrdiff_t *hist,
        double max_estimate[],
        const int nbins)
{
    TIMEZONE("fluid_solver_base::compute_rspace_stats");
    shared_array<double> threaded_local_moments(10*nvals,[&](double* local_moments){
        std::fill_n(local_moments, 10*nvals, 0);
        if (nvals == 4) local_moments[3] = max_estimate[3];
    });

    shared_array<ptrdiff_t> threaded_local_hist(nbins*nvals, [&](ptrdiff_t* local_hist){
        std::fill_n(local_hist, nbins*nvals, 0);
    });

    // Will not be modified by the threads
    double binsize[nvals];
    for (int i=0; i<nvals; i++)
        binsize[i] = 2*max_estimate[i] / nbins;

    RLOOP(
                this,
                [&](ptrdiff_t rindex, ptrdiff_t /*xindex*/, ptrdiff_t /*yindex*/, ptrdiff_t /*zindex*/){
        ptrdiff_t *local_hist = threaded_local_hist.getMine();
        double *local_moments = threaded_local_moments.getMine();

        double val_tmp[nvals];
        if (nvals == 4) val_tmp[3] = 0.0;
        for (int i=0; i<3; i++)
        {
            val_tmp[i] = a[rindex*3+i];
            if (nvals == 4) val_tmp[3] += val_tmp[i]*val_tmp[i];
        }
        if (nvals == 4)
        {
            val_tmp[3] = sqrt(val_tmp[3]);
            if (val_tmp[3] < local_moments[0*nvals+3])
                local_moments[0*nvals+3] = val_tmp[3];
            if (val_tmp[3] > local_moments[9*nvals+3])
                local_moments[9*nvals+3] = val_tmp[3];
            int bin = int(floor(val_tmp[3]*2/binsize[3]));
            if (bin >= 0 && bin < nbins)
                local_hist[bin*nvals+3]++;
        }
        for (int i=0; i<3; i++)
        {
            if (val_tmp[i] < local_moments[0*nvals+i])
                local_moments[0*nvals+i] = val_tmp[i];
            if (val_tmp[i] > local_moments[9*nvals+i])
                local_moments[9*nvals+i] = val_tmp[i];
            int bin = int(floor((val_tmp[i] + max_estimate[i]) / binsize[i]));
            if (bin >= 0 && bin < nbins)
                local_hist[bin*nvals+i]++;
        }
        for (int n=1; n<9; n++){
            double pow_tmp = 1;
            for (int i=0; i<nvals; i++){
                local_moments[n*nvals + i] += (pow_tmp = val_tmp[i]*pow_tmp);
            }
        }
    }
    );

    threaded_local_moments.mergeParallel([&](const int idx, const double& v1, const double& v2) -> double {
          if(nvals == int(4) && idx == 0*nvals+3){
              return std::min(v1, v2);  
          }
          if(nvals == int(4) && idx == 9*nvals+3){
              return std::max(v1, v2);  
          }
          if(idx < 3){
              return std::min(v1, v2);        
          }      
          if(9*nvals <= idx && idx < 9*nvals+3){
              return std::max(v1, v2);        
          }
          return v1 + v2;
      });
    threaded_local_hist.mergeParallel();

    MPI_Allreduce(
                threaded_local_moments.getMasterData(),
                (void*)moments,
                nvals,
                MPI_DOUBLE, MPI_MIN, this->cd->comm);
    MPI_Allreduce(
                (threaded_local_moments.getMasterData() + nvals),
                (void*)(moments+nvals),
                8*nvals,
                MPI_DOUBLE, MPI_SUM, this->cd->comm);
    MPI_Allreduce(
                (threaded_local_moments.getMasterData() + 9*nvals),
                (void*)(moments+9*nvals),
                nvals,
                MPI_DOUBLE, MPI_MAX, this->cd->comm);
    MPI_Allreduce(
                (void*)threaded_local_hist.getMasterData(),
                (void*)hist,
                nbins*nvals,
                MPI_INT64_T, MPI_SUM, this->cd->comm);
    for (int n=1; n<9; n++)
        for (int i=0; i<nvals; i++)
            moments[n*nvals + i] /= this->normalization_factor;
}

template <class rnumber>
void fluid_solver_base<rnumber>::write_spectrum(const char *fname, cnumber *a, const double k2exponent)
{
    TIMEZONE("fluid_solver_base::write_spectrum");
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

template <class rnumber>
fluid_solver_base<rnumber>::fluid_solver_base(
        const char *NAME,
        int nx,
        int ny,
        int nz,
        double DKX,
        double DKY,
        double DKZ,
        int DEALIAS_TYPE,
        unsigned FFTW_PLAN_RIGOR)
{
    TIMEZONE("fluid_solver_base::fluid_solver_base");
    strncpy(this->name, NAME, 256);
    this->name[255] = '\0';
    this->iteration = 0;
    this->fftw_plan_rigor = FFTW_PLAN_RIGOR;

    int ntmp[4];
    ntmp[0] = nz;
    ntmp[1] = ny;
    ntmp[2] = nx;
    ntmp[3] = 3;
    this->rd = new field_descriptor<rnumber>(
                4, ntmp, mpi_real_type<rnumber>::real(), MPI_COMM_WORLD);
    this->normalization_factor = (this->rd->full_size/3);
    ntmp[0] = ny;
    ntmp[1] = nz;
    ntmp[2] = nx/2 + 1;
    ntmp[3] = 3;
    this->cd = new field_descriptor<rnumber>(
                4, ntmp, mpi_real_type<rnumber>::complex(), this->rd->comm);

    this->dkx = DKX;
    this->dky = DKY;
    this->dkz = DKZ;
    this->kx = new double[this->cd->sizes[2]];
    this->ky = new double[this->cd->subsizes[0]];
    this->kz = new double[this->cd->sizes[1]];
    this->dealias_type = DEALIAS_TYPE;
    switch(this->dealias_type)
    {
    /* HL07 smooth filter */
    case 1:
        this->kMx = this->dkx*(int(this->rd->sizes[2] / 2)-1);
        this->kMy = this->dky*(int(this->rd->sizes[1] / 2)-1);
        this->kMz = this->dkz*(int(this->rd->sizes[0] / 2)-1);
        break;
    default:
        this->kMx = this->dkx*(int(this->rd->sizes[2] / 3)-1);
        this->kMy = this->dky*(int(this->rd->sizes[1] / 3)-1);
        this->kMz = this->dkz*(int(this->rd->sizes[0] / 3)-1);
    }
    int i, ii;
    for (i = 0; i<this->cd->sizes[2]; i++)
        this->kx[i] = i*this->dkx;
    for (i = 0; i<this->cd->subsizes[0]; i++)
    {
        ii = i + this->cd->starts[0];
        if (ii <= this->rd->sizes[1]/2)
            this->ky[i] = this->dky*ii;
        else
            this->ky[i] = this->dky*(ii - this->rd->sizes[1]);
    }
    for (i = 0; i<this->cd->sizes[1]; i++)
    {
        if (i <= this->rd->sizes[0]/2)
            this->kz[i] = this->dkz*i;
        else
            this->kz[i] = this->dkz*(i - this->rd->sizes[0]);
    }
    this->kM = this->kMx;
    if (this->kM < this->kMy) this->kM = this->kMy;
    if (this->kM < this->kMz) this->kM = this->kMz;
    this->kM2 = this->kM * this->kM;
    this->kMspec = this->kM;
    this->kMspec2 = this->kM2;
    this->dk = this->dkx;
    if (this->dk > this->dky) this->dk = this->dky;
    if (this->dk > this->dkz) this->dk = this->dkz;
    this->dk2 = this->dk*this->dk;
    DEBUG_MSG(
                "kM = %g, kM2 = %g, dk = %g, dk2 = %g\n",
                this->kM, this->kM2, this->dk, this->dk2);
    /* spectra stuff */
    this->nshells = int(this->kMspec / this->dk) + 2;
    DEBUG_MSG(
                "kMspec = %g, kMspec2 = %g, nshells = %ld\n",
                this->kMspec, this->kMspec2, this->nshells);
    this->kshell = new double[this->nshells];
    std::fill_n(this->kshell, this->nshells, 0.0);
    this->nshell = new int64_t[this->nshells];
    std::fill_n(this->nshell, this->nshells, 0);

    shared_array<double> kshell_local_threaded(this->nshells,[&](double* kshell_local){
        std::fill_n(kshell_local, this->nshells, 0.0);
    });
    shared_array<double> nshell_local_threaded(this->nshells,[&](double* nshell_local){
        std::fill_n(nshell_local, this->nshells, 0.0);
    });

    std::vector<std::unordered_map<int, double>> Fourier_filter_threaded(omp_get_max_threads());

    CLOOP_K2_NXMODES(
                this,

                [&](ptrdiff_t /*cindex*/, ptrdiff_t /*xindex*/, ptrdiff_t /*yindex*/,
                ptrdiff_t /*zindex*/, double k2, int nxmodes){
        if (k2 < this->kM2)
        {
            double knorm = sqrt(k2);
            nshell_local_threaded.getMine()[int(knorm/this->dk)] += nxmodes;
            kshell_local_threaded.getMine()[int(knorm/this->dk)] += nxmodes*knorm;
        }
        Fourier_filter_threaded[omp_get_thread_num()][int(round(k2 / this->dk2))] = exp(-36.0 * pow(k2/this->kM2, 18.));}
    );

    // Merge results
    nshell_local_threaded.mergeParallel();
    kshell_local_threaded.mergeParallel();
    for(int idxMerge = 0 ; idxMerge < int(Fourier_filter_threaded.size()) ; ++idxMerge){
        for(const auto kv : Fourier_filter_threaded[idxMerge]){
            this->Fourier_filter[kv.first] = kv.second;
        }
    }

    MPI_Allreduce(
                (void*)(nshell_local_threaded.getMasterData()),
                (void*)(this->nshell),
                this->nshells,
                MPI_INT64_T, MPI_SUM, this->cd->comm);
    MPI_Allreduce(
                (void*)(kshell_local_threaded.getMasterData()),
                (void*)(this->kshell),
                this->nshells,
                MPI_DOUBLE, MPI_SUM, this->cd->comm);
    for (unsigned int n=0; n<this->nshells; n++)
    {
        this->kshell[n] /= this->nshell[n];
    }
}

template <class rnumber>
fluid_solver_base<rnumber>::~fluid_solver_base()
{
    delete[] this->kshell;
    delete[] this->nshell;

    delete[] this->kx;
    delete[] this->ky;
    delete[] this->kz;

    delete this->cd;
    delete this->rd;
}

template <class rnumber>
void fluid_solver_base<rnumber>::low_pass_Fourier(cnumber *a, const int howmany, const double kmax)
{
    TIMEZONE("fluid_solver_base::low_pass_Fourier");
    const double km2 = kmax*kmax;
    const int howmany2 = 2*howmany;
    /*DEBUG_MSG("entered low_pass_Fourier, kmax=%lg km2=%lg howmany2=%d\n", kmax, km2, howmany2);*/
    CLOOP_K2(
                this,
                /*DEBUG_MSG("kx=%lg ky=%lg kz=%lg k2=%lg\n",
                                  this->kx[xindex],
                                  this->ky[yindex],
                                  this->kz[zindex],
                                  k2);*/

                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex,
                ptrdiff_t zindex, double k2){
        if (k2 >= km2)
            std::fill_n((rnumber*)(a + howmany*cindex), howmany2, 0.0);}
    );
}

template <class rnumber>
void fluid_solver_base<rnumber>::dealias(cnumber *a, const int howmany)
{
    TIMEZONE("fluid_solver_base::dealias");
    if (this->dealias_type == 0)
    {
        this->low_pass_Fourier(a, howmany, this->kM);
        return;
    }

    CLOOP_K2(
                this,
                [&](ptrdiff_t cindex, ptrdiff_t /*xindex*/, ptrdiff_t /*yindex*/,
                ptrdiff_t /*zindex*/, double k2){
        double tval = this->Fourier_filter[int(round(k2/this->dk2))];
        // It is thread safe on the index cindex
        for (int tcounter = 0; tcounter < howmany; tcounter++)
            for (int i=0; i<2; i++)
                a[howmany*cindex+tcounter][i] *= tval;
    }
    );
}

template <class rnumber>
void fluid_solver_base<rnumber>::force_divfree(cnumber *a)
{
    TIMEZONE("fluid_solver_base::force_divfree");
    CLOOP_K2(
                this,

                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex,
                ptrdiff_t zindex, double k2){
        if (k2 > 0)
        {
            // It is thread safe on index cindex
            cnumber tval;
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
        }}
    );
    if (this->cd->myrank == this->cd->rank[0])
        std::fill_n((rnumber*)(a), 6, 0.0);
}

template <class rnumber>
void fluid_solver_base<rnumber>::compute_vector_gradient(cnumber *A, cnumber *cvec)
{
    TIMEZONE("fluid_solver_base::compute_vector_gradient");
    std::fill_n((rnumber*)A, 3*2*this->cd->local_size, 0.0);
    cnumber *dx_u, *dy_u, *dz_u;
    dx_u = A;
    dy_u = A + this->cd->local_size;
    dz_u = A + 2*this->cd->local_size;
    CLOOP_K2(
                this,

                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex,
                ptrdiff_t zindex, double k2){
        if (k2 <= this->kM2)
        {
            // It is thread safe on cindex
            ptrdiff_t tindex = 3*cindex;
            for (int cc=0; cc<3; cc++)
            {
                dx_u[tindex + cc][0] = -this->kx[xindex]*cvec[tindex+cc][1];
                dx_u[tindex + cc][1] =  this->kx[xindex]*cvec[tindex+cc][0];
                dy_u[tindex + cc][0] = -this->ky[yindex]*cvec[tindex+cc][1];
                dy_u[tindex + cc][1] =  this->ky[yindex]*cvec[tindex+cc][0];
                dz_u[tindex + cc][0] = -this->kz[zindex]*cvec[tindex+cc][1];
                dz_u[tindex + cc][1] =  this->kz[zindex]*cvec[tindex+cc][0];
            }
        }}
    );
}

template <class rnumber>
void fluid_solver_base<rnumber>::symmetrize(cnumber *data, const int howmany)
{
    TIMEZONE("fluid_solver_base::symmetrize");
    ptrdiff_t ii, cc;
    MPI_Status *mpistatus = new MPI_Status;
    if (this->cd->myrank == this->cd->rank[0])
    {
        for (cc = 0; cc < howmany; cc++)
            data[cc][1] = 0.0;
        for (ii = 1; ii < this->cd->sizes[1]/2; ii++)
            for (cc = 0; cc < howmany; cc++) {
                ( *(data + cc + howmany*(this->cd->sizes[1] - ii)*this->cd->sizes[2]))[0] =
                        (*(data + cc + howmany*(                     ii)*this->cd->sizes[2]))[0];
                ( *(data + cc + howmany*(this->cd->sizes[1] - ii)*this->cd->sizes[2]))[1] =
                        -(*(data + cc + howmany*(                     ii)*this->cd->sizes[2]))[1];
            }
    }
    cnumber *buffer;
    buffer = fftw_interface<rnumber>::alloc_complex(howmany*this->cd->sizes[1]);
    ptrdiff_t yy;
    /*ptrdiff_t tindex;*/
    int ranksrc, rankdst;
    for (yy = 1; yy < this->cd->sizes[0]/2; yy++) {
        ranksrc = this->cd->rank[yy];
        rankdst = this->cd->rank[this->cd->sizes[0] - yy];
        if (this->cd->myrank == ranksrc)
            for (ii = 0; ii < this->cd->sizes[1]; ii++)
                for (cc = 0; cc < howmany; cc++)
                    for (int imag_comp=0; imag_comp<2; imag_comp++)
                        (*(buffer + howmany*ii+cc))[imag_comp] =
                            (*(data + howmany*((yy - this->cd->starts[0])*this->cd->sizes[1] + ii)*this->cd->sizes[2] + cc))[imag_comp];
        if (ranksrc != rankdst)
        {
            if (this->cd->myrank == ranksrc)
                MPI_Send((void*)buffer,
                         howmany*this->cd->sizes[1], mpi_real_type<rnumber>::complex(), rankdst, yy,
                        this->cd->comm);
            if (this->cd->myrank == rankdst)
                MPI_Recv((void*)buffer,
                         howmany*this->cd->sizes[1], mpi_real_type<rnumber>::complex(), ranksrc, yy,
                        this->cd->comm, mpistatus);
        }
        if (this->cd->myrank == rankdst)
        {
            for (ii = 1; ii < this->cd->sizes[1]; ii++)
                for (cc = 0; cc < howmany; cc++)
                {
                    (*(data + howmany*((this->cd->sizes[0] - yy - this->cd->starts[0])*this->cd->sizes[1] + ii)*this->cd->sizes[2] + cc))[0] =
                            (*(buffer + howmany*(this->cd->sizes[1]-ii)+cc))[0];
                    (*(data + howmany*((this->cd->sizes[0] - yy - this->cd->starts[0])*this->cd->sizes[1] + ii)*this->cd->sizes[2] + cc))[1] =
                            -(*(buffer + howmany*(this->cd->sizes[1]-ii)+cc))[1];
                }
            for (cc = 0; cc < howmany; cc++)
            {
                (*((data + cc + howmany*(this->cd->sizes[0] - yy - this->cd->starts[0])*this->cd->sizes[1]*this->cd->sizes[2])))[0] =  (*(buffer + cc))[0];
                (*((data + cc + howmany*(this->cd->sizes[0] - yy - this->cd->starts[0])*this->cd->sizes[1]*this->cd->sizes[2])))[1] = -(*(buffer + cc))[1];
            }
        }
    }
    fftw_interface<rnumber>::free(buffer);
    delete mpistatus;
    /* put asymmetric data to 0 */
    /*if (this->cd->myrank == this->cd->rank[this->cd->sizes[0]/2])
    {
        tindex = howmany*(this->cd->sizes[0]/2 - this->cd->starts[0])*this->cd->sizes[1]*this->cd->sizes[2];
        for (ii = 0; ii < this->cd->sizes[1]; ii++)
        {
            std::fill_n((rnumber*)(data + tindex), howmany*2*this->cd->sizes[2], 0.0);
            tindex += howmany*this->cd->sizes[2];
        }
    }
    tindex = howmany*();
    std::fill_n((rnumber*)(data + tindex), howmany*2, 0.0);*/
}

template <class rnumber>
int fluid_solver_base<rnumber>::read_base(const char *fname, rnumber *data)
{
    char full_name[512];
    sprintf(full_name, "%s_%s_i%.5x", this->name, fname, this->iteration);
    return this->rd->read(full_name, (void*)data);
}

template <class rnumber>
int fluid_solver_base<rnumber>::read_base(const char *fname, cnumber *data)
{
    char full_name[512];
    sprintf(full_name, "%s_%s_i%.5x", this->name, fname, this->iteration);
    return this->cd->read(full_name, (void*)data);
}

template <class rnumber>
int fluid_solver_base<rnumber>::write_base(const char *fname, rnumber *data)
{
    char full_name[512];
    sprintf(full_name, "%s_%s_i%.5x", this->name, fname, this->iteration);
    return this->rd->write(full_name, (void*)data);
}

template <class rnumber>
int fluid_solver_base<rnumber>::write_base(const char *fname, cnumber *data)
{
    char full_name[512];
    sprintf(full_name, "%s_%s_i%.5x", this->name, fname, this->iteration);
    return this->cd->write(full_name, (void*)data);
}

/* finally, force generation of code                                         */
template class fluid_solver_base<float>;
template class fluid_solver_base<double>;

/*****************************************************************************/




