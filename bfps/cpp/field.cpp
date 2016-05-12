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


#include <cstdlib>
#include <algorithm>
#include <cassert>
#include "field.hpp"

template <field_components fc>
field_layout<fc>::field_layout(
        const hsize_t *SIZES,
        const hsize_t *SUBSIZES,
        const hsize_t *STARTS,
        const MPI_Comm COMM_TO_USE)
{
    this->comm = COMM_TO_USE;
    MPI_Comm_rank(this->comm, &this->myrank);
    MPI_Comm_size(this->comm, &this->nprocs);

    std::copy(SIZES, SIZES + 3, this->sizes);
    std::copy(SUBSIZES, SUBSIZES + 3, this->subsizes);
    std::copy(STARTS, STARTS + 3, this->starts);
    if (fc == THREE || fc == THREExTHREE)
    {
        this->sizes[3] = 3;
        this->subsizes[3] = 3;
        this->starts[3] = 0;
    }
    if (fc == THREExTHREE)
    {
        this->sizes[4] = 3;
        this->subsizes[4] = 3;
        this->starts[4] = 0;
    }
    this->local_size = 1;
    this->full_size = 1;
    for (unsigned int i=0; i<ndim(fc); i++)
    {
        this->local_size *= this->subsizes[i];
        this->full_size *= this->sizes[i];
    }

    /*field will at most be distributed in 2D*/
    this->rank.resize(2);
    this->all_start.resize(2);
    this->all_size.resize(2);
    for (int i=0; i<2; i++)
    {
        this->rank[i].resize(this->sizes[i]);
        std::vector<int> local_rank;
        local_rank.resize(this->sizes[i], 0);
        for (unsigned int ii=this->starts[i]; ii<this->starts[i]+this->subsizes[i]; ii++)
            local_rank[ii] = this->myrank;
        MPI_Allreduce(
                &local_rank.front(),
                &this->rank[i].front(),
                this->sizes[i],
                MPI_INT,
                MPI_SUM,
                this->comm);
        this->all_start[i].resize(this->nprocs);
        std::vector<int> local_start;
        local_start.resize(this->nprocs, 0);
        local_start[this->myrank] = this->starts[i];
        MPI_Allreduce(
                &local_start.front(),
                &this->all_start[i].front(),
                this->nprocs,
                MPI_INT,
                MPI_SUM,
                this->comm);
        this->all_size[i].resize(this->nprocs);
        std::vector<int> local_subsize;
        local_subsize.resize(this->nprocs, 0);
        local_subsize[this->myrank] = this->subsizes[i];
        MPI_Allreduce(
                &local_subsize.front(),
                &this->all_size[i].front(),
                this->nprocs,
                MPI_INT,
                MPI_SUM,
                this->comm);
    }
}

template <typename rnumber,
          field_backend be,
          field_components fc>
field<rnumber, be, fc>::field(
                const int nx,
                const int ny,
                const int nz,
                const MPI_Comm COMM_TO_USE,
                const unsigned FFTW_PLAN_RIGOR)
{
    this->comm = COMM_TO_USE;
    MPI_Comm_rank(this->comm, &this->myrank);
    MPI_Comm_size(this->comm, &this->nprocs);

    this->fftw_plan_rigor = FFTW_PLAN_RIGOR;
    this->real_space_representation = true;

    /* generate HDF5 data types */
    if (typeid(rnumber) == typeid(float))
        this->rnumber_H5T = H5Tcopy(H5T_NATIVE_FLOAT);
    else if (typeid(rnumber) == typeid(double))
        this->rnumber_H5T = H5Tcopy(H5T_NATIVE_DOUBLE);
    typedef struct {
        rnumber re;   /*real part*/
        rnumber im;   /*imaginary part*/
    } tmp_complex_type;
    this->cnumber_H5T = H5Tcreate(H5T_COMPOUND, sizeof(tmp_complex_type));
    H5Tinsert(this->cnumber_H5T, "r", HOFFSET(tmp_complex_type, re), this->rnumber_H5T);
    H5Tinsert(this->cnumber_H5T, "i", HOFFSET(tmp_complex_type, im), this->rnumber_H5T);

    /* switch on backend */
    switch(be)
    {
        case FFTW:
            ptrdiff_t nfftw[3];
            nfftw[0] = nz;
            nfftw[1] = ny;
            nfftw[2] = nx;
            //ptrdiff_t tmp_local_size;
            ptrdiff_t local_n0, local_0_start;
            ptrdiff_t local_n1, local_1_start;
            //tmp_local_size = fftw_mpi_local_size_many_transposed(
            fftw_mpi_local_size_many_transposed(
                    3, nfftw, ncomp(fc),
                    FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, this->comm,
                    &local_n0, &local_0_start,
                    &local_n1, &local_1_start);
            hsize_t sizes[3], subsizes[3], starts[3];
            sizes[0] = nz; sizes[1] = ny; sizes[2] = nx;
            subsizes[0] = local_n0; subsizes[1] = ny; subsizes[2] = nx;
            starts[0] = local_0_start; starts[1] = 0; starts[2] = 0;
            this->rlayout = new field_layout<fc>(
                    sizes, subsizes, starts, this->comm);
            this->npoints = this->rlayout->full_size / ncomp(fc);
            sizes[0] = nz; sizes[1] = ny; sizes[2] = nx+2;
            subsizes[0] = local_n0; subsizes[1] = ny; subsizes[2] = nx+2;
            starts[0] = local_0_start; starts[1] = 0; starts[2] = 0;
            this->rmemlayout = new field_layout<fc>(
                    sizes, subsizes, starts, this->comm);
            sizes[0] = nz; sizes[1] = ny; sizes[2] = nx/2+1;
            subsizes[0] = local_n1; subsizes[1] = ny; subsizes[2] = nx/2+1;
            starts[0] = local_1_start; starts[1] = 0; starts[2] = 0;
            this->clayout = new field_layout<fc>(
                    sizes, subsizes, starts, this->comm);
            this->data = (rnumber*)fftw_malloc(
                    sizeof(rnumber)*this->rmemlayout->local_size);
            if(typeid(rnumber) == typeid(float))
            {
                this->c2r_plan = new fftwf_plan;
                this->r2c_plan = new fftwf_plan;
                *((fftwf_plan*)this->c2r_plan) = fftwf_mpi_plan_many_dft_c2r(
                        3, nfftw, ncomp(fc),
                        FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                        (fftwf_complex*)this->data, (float*)this->data,
                        this->comm,
                        this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_IN);
                *((fftwf_plan*)this->r2c_plan) = fftwf_mpi_plan_many_dft_r2c(
                        3, nfftw, ncomp(fc),
                        FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                        (float*)this->data, (fftwf_complex*)this->data,
                        this->comm,
                        this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_OUT);
            }
            if (typeid(rnumber) == typeid(double))
            {
                this->c2r_plan = new fftw_plan;
                this->r2c_plan = new fftw_plan;
                *((fftw_plan*)this->c2r_plan) = fftw_mpi_plan_many_dft_c2r(
                        3, nfftw, ncomp(fc),
                        FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                        (fftw_complex*)this->data, (double*)this->data,
                        this->comm,
                        this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_IN);
                *((fftw_plan*)this->r2c_plan) = fftw_mpi_plan_many_dft_r2c(
                        3, nfftw, ncomp(fc),
                        FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                        (double*)this->data, (fftw_complex*)this->data,
                        this->comm,
                        this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_OUT);
            }
            break;
    }
}

template <typename rnumber,
          field_backend be,
          field_components fc>
field<rnumber, be, fc>::~field()
{
    /* close data types */
    H5Tclose(this->rnumber_H5T);
    H5Tclose(this->cnumber_H5T);
    switch(be)
    {
        case FFTW:
            delete this->rlayout;
            delete this->rmemlayout;
            delete this->clayout;
            fftw_free(this->data);
            if (typeid(rnumber) == typeid(float))
            {
                fftwf_destroy_plan(*(fftwf_plan*)this->c2r_plan);
                delete (fftwf_plan*)this->c2r_plan;
                fftwf_destroy_plan(*(fftwf_plan*)this->r2c_plan);
                delete (fftwf_plan*)this->r2c_plan;
            }
            else if (typeid(rnumber) == typeid(double))
            {
                fftw_destroy_plan(*(fftw_plan*)this->c2r_plan);
                delete (fftw_plan*)this->c2r_plan;
                fftw_destroy_plan(*(fftw_plan*)this->r2c_plan);
                delete (fftw_plan*)this->r2c_plan;
            }
            break;
    }
}

template <typename rnumber,
          field_backend be,
          field_components fc>
void field<rnumber, be, fc>::ift()
{
    if (typeid(rnumber) == typeid(float))
        fftwf_execute(*((fftwf_plan*)this->c2r_plan));
    else if (typeid(rnumber) == typeid(double))
        fftw_execute(*((fftw_plan*)this->c2r_plan));
    this->real_space_representation = true;
}

template <typename rnumber,
          field_backend be,
          field_components fc>
void field<rnumber, be, fc>::dft()
{
    if (typeid(rnumber) == typeid(float))
        fftwf_execute(*((fftwf_plan*)this->r2c_plan));
    else if (typeid(rnumber) == typeid(double))
        fftw_execute(*((fftw_plan*)this->r2c_plan));
    this->real_space_representation = false;
}

template <typename rnumber,
          field_backend be,
          field_components fc>
int field<rnumber, be, fc>::io(
        const std::string fname,
        const std::string dset_name,
        const int toffset,
        const bool read)
{
    hid_t file_id, dset_id, plist_id;
    hid_t dset_type;
    bool io_for_real = false;

    /* open file */
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, this->comm, MPI_INFO_NULL);
    if (read)
        file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, plist_id);
    else
        file_id = H5Fopen(fname.c_str(), H5F_ACC_RDWR, plist_id);
    H5Pclose(plist_id);

    /* open data set */
    dset_id = H5Dopen(file_id, dset_name.c_str(), H5P_DEFAULT);
    dset_type = H5Dget_type(dset_id);
    io_for_real = (
            H5Tequal(dset_type, H5T_IEEE_F32BE) ||
            H5Tequal(dset_type, H5T_IEEE_F32LE) ||
            H5Tequal(dset_type, H5T_INTEL_F32) ||
            H5Tequal(dset_type, H5T_NATIVE_FLOAT) ||
            H5Tequal(dset_type, H5T_IEEE_F64BE) ||
            H5Tequal(dset_type, H5T_IEEE_F64LE) ||
            H5Tequal(dset_type, H5T_INTEL_F64) ||
            H5Tequal(dset_type, H5T_NATIVE_DOUBLE));

    /* generic space initialization */
    hid_t fspace, mspace;
    fspace = H5Dget_space(dset_id);
    hsize_t count[ndim(fc)+1], offset[ndim(fc)+1], dims[ndim(fc)+1];
    hsize_t memoffset[ndim(fc)+1], memshape[ndim(fc)+1];
    H5Sget_simple_extent_dims(fspace, dims, NULL);
    count[0] = 1;
    offset[0] = toffset;
    memshape[0] = 1;
    memoffset[0] = 0;
    if (io_for_real)
    {
        for (unsigned int i=0; i<ndim(fc); i++)
        {
            count[i+1] = this->rlayout->subsizes[i];
            offset[i+1] = this->rlayout->starts[i];
            assert(dims[i+1] == this->rlayout->sizes[i]);
            memshape[i+1] = this->rmemlayout->subsizes[i];
            memoffset[i+1] = 0;
        }
        mspace = H5Screate_simple(ndim(fc)+1, memshape, NULL);
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Sselect_hyperslab(mspace, H5S_SELECT_SET, memoffset, NULL, count, NULL);
        if (read)
        {
            std::fill_n(this->data, this->rmemlayout->local_size, 0);
            H5Dread(dset_id, this->rnumber_H5T, mspace, fspace, H5P_DEFAULT, this->data);
            this->real_space_representation = true;
        }
        else
        {
            H5Dwrite(dset_id, this->rnumber_H5T, mspace, fspace, H5P_DEFAULT, this->data);
            if (!this->real_space_representation)
                /* in principle we could do an inverse Fourier transform in here,
                 * however that would be unsafe since we wouldn't know whether we'd need to
                 * normalize or not.
                 * */
                DEBUG_MSG("I just wrote complex field into real space dataset. It's probably nonsense.\n");
        }
        H5Sclose(mspace);
    }
    else
    {
        for (unsigned int i=0; i<ndim(fc); i++)
        {
            count[i+1] = this->clayout->subsizes[i];
            offset[i+1] = this->clayout->starts[i];
            assert(dims[i+1] == this->clayout->sizes[i]);
            memshape[i+1] = count[i+1];
            memoffset[i+1] = 0;
        }
        mspace = H5Screate_simple(ndim(fc)+1, memshape, NULL);
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Sselect_hyperslab(mspace, H5S_SELECT_SET, memoffset, NULL, count, NULL);
        if (read)
        {
            H5Dread(dset_id, this->cnumber_H5T, mspace, fspace, H5P_DEFAULT, this->data);
            this->real_space_representation = false;
        }
        else
        {
            H5Dwrite(dset_id, this->cnumber_H5T, mspace, fspace, H5P_DEFAULT, this->data);
            if (this->real_space_representation)
                DEBUG_MSG("I just wrote real space field into complex dataset. It's probably nonsense.\n");
        }
        H5Sclose(mspace);
    }

    H5Tclose(dset_type);
    H5Sclose(fspace);
    /* close data set */
    H5Dclose(dset_id);
    /* close file */
    H5Fclose(file_id);
    return EXIT_SUCCESS;
}

template <typename rnumber,
          field_backend be,
          field_components fc>
void field<rnumber, be, fc>::compute_rspace_stats(
                const hid_t group,
                const std::string dset_name,
                const hsize_t toffset,
                const std::vector<double> max_estimate)
{
    assert(this->real_space_representation);
    assert(fc == ONE || fc == THREE);
    const unsigned int nmoments = 10;
    int nvals, nbins;
    if (this->myrank == 0)
    {
        hid_t dset, wspace;
        hsize_t dims[ndim(fc)-1];
        int ndims;
        dset = H5Dopen(group, ("moments/" + dset_name).c_str(), H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        ndims = H5Sget_simple_extent_dims(wspace, dims, NULL);
        assert(ndims == int(ndim(fc))-1);
        assert(dims[1] == nmoments);
        switch(ndims)
        {
            case 2:
                nvals = 1;
                break;
            case 3:
                nvals = dims[2];
                break;
            case 4:
                nvals = dims[2]*dims[3];
                break;
        }
        H5Sclose(wspace);
        H5Dclose(dset);
        dset = H5Dopen(group, ("histograms/" + dset_name).c_str(), H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        ndims = H5Sget_simple_extent_dims(wspace, dims, NULL);
        assert(ndims == int(ndim(fc))-1);
        nbins = dims[1];
        if (ndims == 3)
            assert(nvals == int(dims[2]));
        else if (ndims == 4)
            assert(nvals == int(dims[2]*dims[3]));
        H5Sclose(wspace);
        H5Dclose(dset);
    }
    MPI_Bcast(&nvals, 1, MPI_INT, 0, this->comm);
    MPI_Bcast(&nbins, 1, MPI_INT, 0, this->comm);
    assert(nvals == int(max_estimate.size()));
    double *moments = new double[nmoments*nvals];
    double *local_moments = new double[nmoments*nvals];
    double *val_tmp = new double[nvals];
    double *binsize = new double[nvals];
    double *pow_tmp = new double[nvals];
    ptrdiff_t *hist = new ptrdiff_t[nbins*nvals];
    ptrdiff_t *local_hist = new ptrdiff_t[nbins*nvals];
    int bin;
    for (int i=0; i<nvals; i++)
        binsize[i] = 2*max_estimate[i] / nbins;
    std::fill_n(local_hist, nbins*nvals, 0);
    std::fill_n(local_moments, nmoments*nvals, 0);
    if (nvals == 4) local_moments[3] = max_estimate[3];
    FIELD_RLOOP(
            this,
            std::fill_n(pow_tmp, nvals, 1.0);
            if (nvals == int(4)) val_tmp[3] = 0.0;
            for (unsigned int i=0; i<ncomp(fc); i++)
            {
                val_tmp[i] = this->data[rindex*ncomp(fc)+i];
                if (nvals == int(4)) val_tmp[3] += val_tmp[i]*val_tmp[i];
            }
            if (nvals == int(4))
            {
                val_tmp[3] = sqrt(val_tmp[3]);
                if (val_tmp[3] < local_moments[0*nvals+3])
                    local_moments[0*nvals+3] = val_tmp[3];
                if (val_tmp[3] > local_moments[9*nvals+3])
                    local_moments[9*nvals+3] = val_tmp[3];
                bin = int(floor(val_tmp[3]*2/binsize[3]));
                if (bin >= 0 && bin < nbins)
                    local_hist[bin*nvals+3]++;
            }
            for (unsigned int i=0; i<ncomp(fc); i++)
            {
                if (val_tmp[i] < local_moments[0*nvals+i])
                    local_moments[0*nvals+i] = val_tmp[i];
                if (val_tmp[i] > local_moments[(nmoments-1)*nvals+i])
                    local_moments[(nmoments-1)*nvals+i] = val_tmp[i];
                bin = int(floor((val_tmp[i] + max_estimate[i]) / binsize[i]));
                if (bin >= 0 && bin < nbins)
                    local_hist[bin*nvals+i]++;
            }
            for (int n=1; n < int(nmoments)-1; n++)
                for (int i=0; i<nvals; i++)
                    local_moments[n*nvals + i] += (pow_tmp[i] = val_tmp[i]*pow_tmp[i]);
            );
    MPI_Allreduce(
            (void*)local_moments,
            (void*)moments,
            nvals,
            MPI_DOUBLE, MPI_MIN, this->comm);
    MPI_Allreduce(
            (void*)(local_moments + nvals),
            (void*)(moments+nvals),
            (nmoments-2)*nvals,
            MPI_DOUBLE, MPI_SUM, this->comm);
    MPI_Allreduce(
            (void*)(local_moments + (nmoments-1)*nvals),
            (void*)(moments+(nmoments-1)*nvals),
            nvals,
            MPI_DOUBLE, MPI_MAX, this->comm);
    MPI_Allreduce(
            (void*)local_hist,
            (void*)hist,
            nbins*nvals,
            MPI_INT64_T, MPI_SUM, this->comm);
    for (int n=1; n < int(nmoments)-1; n++)
        for (int i=0; i<nvals; i++)
            moments[n*nvals + i] /= this->npoints;
    delete[] local_moments;
    delete[] local_hist;
    delete[] val_tmp;
    delete[] binsize;
    delete[] pow_tmp;
    if (this->myrank == 0)
    {
        hid_t dset, wspace, mspace;
        hsize_t count[ndim(fc)-1], offset[ndim(fc)-1], dims[ndim(fc)-1];
        dset = H5Dopen(group, ("moments/" + dset_name).c_str(), H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        H5Sget_simple_extent_dims(wspace, dims, NULL);
        offset[0] = toffset;
        offset[1] = 0;
        count[0] = 1;
        count[1] = nmoments;
        if (fc == THREE)
        {
            offset[2] = 0;
            count[2] = nvals;
        }
        if (fc == THREExTHREE)
        {
            offset[2] = 0;
            count[2] = 3;
            offset[3] = 0;
            count[3] = 3;
        }
        mspace = H5Screate_simple(ndim(fc)-1, count, NULL);
        H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, wspace, H5P_DEFAULT, moments);
        H5Sclose(wspace);
        H5Sclose(mspace);
        H5Dclose(dset);
        dset = H5Dopen(group, ("histograms/" + dset_name).c_str(), H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        count[1] = nbins;
        mspace = H5Screate_simple(ndim(fc)-1, count, NULL);
        H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Dwrite(dset, H5T_NATIVE_INT64, mspace, wspace, H5P_DEFAULT, hist);
        H5Sclose(wspace);
        H5Sclose(mspace);
        H5Dclose(dset);
    }
    delete[] moments;
    delete[] hist;
}

template <typename rnumber,
          field_backend be,
          field_components fc>
template <kspace_dealias_type dt>
void field<rnumber, be, fc>::compute_stats(
        kspace<be, dt> *kk,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset,
        const double max_estimate)
{
    std::vector<double> max_estimate_vector;
    bool did_rspace = false;
    switch(fc)
    {
        case ONE:
            max_estimate_vector.resize(1, max_estimate);
            break;
        case THREE:
            max_estimate_vector.resize(4, max_estimate);
            max_estimate_vector[3] *= sqrt(3);
            break;
        case THREExTHREE:
            max_estimate_vector.resize(9, max_estimate);
            break;
    }
    if (this->real_space_representation)
    {
        this->compute_rspace_stats(
                group,
                dset_name,
                toffset,
                max_estimate_vector);
        did_rspace = true;
        this->dft();
        // normalize
        for (hsize_t tmp_index=0; tmp_index<this->rmemlayout->local_size; tmp_index++)
            this->data[tmp_index] /= this->npoints;
    }
    // what follows gave me a headache until I found this link:
    // http://stackoverflow.com/questions/8256636/expected-primary-expression-error-on-template-method-using
    kk->template cospectrum<rnumber, fc>(
            this->get_cdata(),
            this->get_cdata(),
            group,
            dset_name + "_" + dset_name,
            toffset);
    if (!did_rspace)
    {
        this->ift();
        // normalization not required
        this->compute_rspace_stats(
                group,
                dset_name,
                toffset,
                max_estimate_vector);
    }
}

template <field_backend be,
          kspace_dealias_type dt>
template <field_components fc>
kspace<be, dt>::kspace(
        const field_layout<fc> *source_layout,
        const double DKX,
        const double DKY,
        const double DKZ)
{
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
            for (i = 0; i<this->layout->sizes[2]; i++)
                this->kx[i] = i*this->dkx;
            for (i = 0; i<this->layout->subsizes[0]; i++)
            {
                ii = i + this->layout->starts[0];
                if (ii <= this->layout->sizes[1]/2)
                    this->ky[i] = this->dky*ii;
                else
                    this->ky[i] = this->dky*(ii - int(this->layout->sizes[1]));
            }
            for (i = 0; i<this->layout->sizes[1]; i++)
            {
                if (i <= this->layout->sizes[0]/2)
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
    double knorm;
    KSPACE_CLOOP_K2_NXMODES(
            this,
            if (k2 < this->kM2)
            {
                knorm = sqrt(k2);
                nshell_local[int(knorm/this->dk)] += nxmodes;
                kshell_local[int(knorm/this->dk)] += nxmodes*knorm;
            }
            if (dt == TWO_THIRDS)
                this->dealias_filter[int(round(k2 / this->dk2))] = exp(-36.0 * pow(k2/this->kM2, 18.));
            );
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
    KSPACE_CLOOP_K2(
            this,
            if (k2 >= km2)
                std::fill_n(a + 2*ncomp(fc)*cindex, 2*ncomp(fc), 0);
            );
}

template <field_backend be,
          kspace_dealias_type dt>
template <typename rnumber,
          field_components fc>
void kspace<be, dt>::dealias(rnumber *__restrict__ a)
{
    switch(be)
    {
        case TWO_THIRDS:
            this->low_pass<rnumber, fc>(a, this->kM);
            break;
        case SMOOTH:
            KSPACE_CLOOP_K2(
                    this,
                    double tval = this->dealias_filter[int(round(k2 / this->dk2))];
                    for (int tcounter=0; tcounter<2*ncomp(fc); tcounter++)
                        a[2*ncomp(fc)*cindex + tcounter] *= tval;
                    );
            break;
    }
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
    std::vector<double> spec, spec_local;
    spec.resize(this->nshells*ncomp(fc)*ncomp(fc), 0);
    spec_local.resize(this->nshells*ncomp(fc)*ncomp(fc), 0);
    KSPACE_CLOOP_K2_NXMODES(
            this,
            if (k2 <= this->kM2)
            {
                int tmp_int = int(sqrt(k2) / this->dk)*ncomp(fc)*ncomp(fc);
                for (int i=0; i<ncomp(fc); i++)
                for (int j=0; j<ncomp(fc); j++)
                    spec_local[tmp_int + i*ncomp(fc)+j] += nxmodes * (
                    (a[ncomp(fc)*cindex + i][0] * b[ncomp(fc)*cindex + j][0]) +
                    (a[ncomp(fc)*cindex + i][1] * b[ncomp(fc)*cindex + j][1]));
            }
            );
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

template class field<float, FFTW, ONE>;
template class field<float, FFTW, THREE>;
template class field<float, FFTW, THREExTHREE>;
template class field<double, FFTW, ONE>;
template class field<double, FFTW, THREE>;
template class field<double, FFTW, THREExTHREE>;

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

template void field<float, FFTW, ONE>::compute_stats<TWO_THIRDS>(
        kspace<FFTW, TWO_THIRDS> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<float, FFTW, THREE>::compute_stats<TWO_THIRDS>(
        kspace<FFTW, TWO_THIRDS> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<float, FFTW, THREExTHREE>::compute_stats<TWO_THIRDS>(
        kspace<FFTW, TWO_THIRDS> *,
        const hid_t, const std::string, const hsize_t, const double);

template void field<double, FFTW, ONE>::compute_stats<TWO_THIRDS>(
        kspace<FFTW, TWO_THIRDS> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<double, FFTW, THREE>::compute_stats<TWO_THIRDS>(
        kspace<FFTW, TWO_THIRDS> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<double, FFTW, THREExTHREE>::compute_stats<TWO_THIRDS>(
        kspace<FFTW, TWO_THIRDS> *,
        const hid_t, const std::string, const hsize_t, const double);

template void field<float, FFTW, ONE>::compute_stats<SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<float, FFTW, THREE>::compute_stats<SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<float, FFTW, THREExTHREE>::compute_stats<SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        const hid_t, const std::string, const hsize_t, const double);

template void field<double, FFTW, ONE>::compute_stats<SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<double, FFTW, THREE>::compute_stats<SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<double, FFTW, THREExTHREE>::compute_stats<SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        const hid_t, const std::string, const hsize_t, const double);

