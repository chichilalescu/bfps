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



//#define NDEBUG

#include <cassert>
#include <cmath>
#include <cstring>
#include "fluid_solver.hpp"
#include "fftw_tools.hpp"
#include "scope_timer.hpp"



template <class rnumber>
void fluid_solver<rnumber>::impose_zero_modes()
{
    if (this->cd->myrank == this->cd->rank[0])
    {
        std::fill_n((rnumber*)(this->cu), 6, 0.0);
        std::fill_n((rnumber*)(this->cv[0]), 6, 0.0);
        std::fill_n((rnumber*)(this->cv[1]), 6, 0.0);
        std::fill_n((rnumber*)(this->cv[2]), 6, 0.0);
    }
}
/*****************************************************************************/
/* macro for specializations to numeric types compatible with FFTW           */

template <class rnumber>
fluid_solver<rnumber>::fluid_solver(
        const char *NAME,
        int nx,
        int ny,
        int nz,
        double DKX,
        double DKY,
        double DKZ,
        int DEALIAS_TYPE,
        unsigned FFTW_PLAN_RIGOR) : fluid_solver_base<rnumber>(
                                        NAME,
                                        nx , ny , nz,
                                        DKX, DKY, DKZ,
                                        DEALIAS_TYPE,
                                        FFTW_PLAN_RIGOR)
{
    TIMEZONE("fluid_solver::fluid_solver");
    this->cvorticity = fftw_interface<rnumber>::alloc_complex(this->cd->local_size);
    this->cvelocity  = fftw_interface<rnumber>::alloc_complex(this->cd->local_size);
    this->rvorticity = fftw_interface<rnumber>::alloc_real(this->cd->local_size*2);
    /*this->rvelocity  = (rnumber*)(this->cvelocity);*/
    this->rvelocity  = fftw_interface<rnumber>::alloc_real(this->cd->local_size*2);

    this->ru = this->rvelocity;
    this->cu = this->cvelocity;

    this->rv[0] = this->rvorticity;
    this->rv[3] = this->rvorticity;
    this->cv[0] = this->cvorticity;
    this->cv[3] = this->cvorticity;

    this->cv[1] = fftw_interface<rnumber>::alloc_complex(this->cd->local_size);
    this->cv[2] = this->cv[1];
    this->rv[1] = fftw_interface<rnumber>::alloc_real(this->cd->local_size*2);
    this->rv[2] = this->rv[1];

    this->c2r_vorticity = new typename fftw_interface<rnumber>::plan;
    this->r2c_vorticity = new typename fftw_interface<rnumber>::plan;
    this->c2r_velocity  = new typename fftw_interface<rnumber>::plan;
    this->r2c_velocity  = new typename fftw_interface<rnumber>::plan;

    ptrdiff_t sizes[] = {nz,
                         ny,
                         nx};

    *this->c2r_vorticity = fftw_interface<rnumber>::mpi_plan_many_dft_c2r(
                3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                this->cvorticity, this->rvorticity,
                MPI_COMM_WORLD, this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_IN);

    *this->r2c_vorticity = fftw_interface<rnumber>::mpi_plan_many_dft_r2c(
                3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                this->rvorticity, this->cvorticity,
                MPI_COMM_WORLD, this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_OUT);

    *this->c2r_velocity = fftw_interface<rnumber>::mpi_plan_many_dft_c2r(
                3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                this->cvelocity, this->rvelocity,
                MPI_COMM_WORLD, this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_IN);

    *this->r2c_velocity = fftw_interface<rnumber>::mpi_plan_many_dft_r2c(
                3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                this->rvelocity, this->cvelocity,
                MPI_COMM_WORLD, this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_OUT);

    this->uc2r = this->c2r_velocity;
    this->ur2c = this->r2c_velocity;
    this->vc2r[0] = this->c2r_vorticity;
    this->vr2c[0] = this->r2c_vorticity;

    this->vc2r[1] = new typename fftw_interface<rnumber>::plan;
    this->vr2c[1] = new typename fftw_interface<rnumber>::plan;
    this->vc2r[2] = new typename fftw_interface<rnumber>::plan;
    this->vr2c[2] = new typename fftw_interface<rnumber>::plan;

    *(this->vc2r[1]) = fftw_interface<rnumber>::mpi_plan_many_dft_c2r(
                3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                this->cv[1], this->rv[1],
            MPI_COMM_WORLD, this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_IN);

    *this->vc2r[2] = fftw_interface<rnumber>::mpi_plan_many_dft_c2r(
                3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                this->cv[2], this->rv[2],
            MPI_COMM_WORLD, this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_IN);

    *this->vr2c[1] = fftw_interface<rnumber>::mpi_plan_many_dft_r2c(
                3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                this->rv[1], this->cv[1],
            MPI_COMM_WORLD, this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_OUT);

    *this->vr2c[2] = fftw_interface<rnumber>::mpi_plan_many_dft_r2c(
                3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                this->rv[2], this->cv[2],
            MPI_COMM_WORLD, this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_OUT);

    /* ``physical'' parameters etc, initialized here just in case */

    this->nu = 0.1;
    this->fmode = 1;
    this->famplitude = 1.0;
    this->fk0  = 0;
    this->fk1 = 3.0;
    /* initialization of fields must be done AFTER planning */
    std::fill_n((rnumber*)this->cvorticity, this->cd->local_size*2, 0.0);
    std::fill_n((rnumber*)this->cvelocity, this->cd->local_size*2, 0.0);
    std::fill_n(this->rvelocity, this->cd->local_size*2, 0.0);
    std::fill_n(this->rvorticity, this->cd->local_size*2, 0.0);
    std::fill_n((rnumber*)this->cv[1], this->cd->local_size*2, 0.0);
    std::fill_n(this->rv[1], this->cd->local_size*2, 0.0);
    std::fill_n(this->rv[2], this->cd->local_size*2, 0.0);
}

template <class rnumber>
fluid_solver<rnumber>::~fluid_solver()
{
    fftw_interface<rnumber>::destroy_plan(*this->c2r_vorticity);
    fftw_interface<rnumber>::destroy_plan(*this->r2c_vorticity);
    fftw_interface<rnumber>::destroy_plan(*this->c2r_velocity );
    fftw_interface<rnumber>::destroy_plan(*this->r2c_velocity );
    fftw_interface<rnumber>::destroy_plan(*this->vc2r[1]);
    fftw_interface<rnumber>::destroy_plan(*this->vr2c[1]);
    fftw_interface<rnumber>::destroy_plan(*this->vc2r[2]);
    fftw_interface<rnumber>::destroy_plan(*this->vr2c[2]);

    delete this->c2r_vorticity;
    delete this->r2c_vorticity;
    delete this->c2r_velocity ;
    delete this->r2c_velocity ;
    delete this->vc2r[1];
    delete this->vr2c[1];
    delete this->vc2r[2];
    delete this->vr2c[2];

    fftw_interface<rnumber>::free(this->cv[1]);
    fftw_interface<rnumber>::free(this->rv[1]);
    fftw_interface<rnumber>::free(this->cvorticity);
    fftw_interface<rnumber>::free(this->rvorticity);
    fftw_interface<rnumber>::free(this->cvelocity);
    fftw_interface<rnumber>::free(this->rvelocity);
}

template <class rnumber>
void fluid_solver<rnumber>::compute_vorticity()
{
    TIMEZONE("fluid_solver::compute_vorticity");
    ptrdiff_t tindex;
    CLOOP_K2(
                this,
                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
        tindex = 3*cindex;
        if (k2 <= this->kM2)
        {
            this->cvorticity[tindex+0][0] = -(this->ky[yindex]*this->cu[tindex+2][1] - this->kz[zindex]*this->cu[tindex+1][1]);
            this->cvorticity[tindex+1][0] = -(this->kz[zindex]*this->cu[tindex+0][1] - this->kx[xindex]*this->cu[tindex+2][1]);
            this->cvorticity[tindex+2][0] = -(this->kx[xindex]*this->cu[tindex+1][1] - this->ky[yindex]*this->cu[tindex+0][1]);
            this->cvorticity[tindex+0][1] =  (this->ky[yindex]*this->cu[tindex+2][0] - this->kz[zindex]*this->cu[tindex+1][0]);
            this->cvorticity[tindex+1][1] =  (this->kz[zindex]*this->cu[tindex+0][0] - this->kx[xindex]*this->cu[tindex+2][0]);
            this->cvorticity[tindex+2][1] =  (this->kx[xindex]*this->cu[tindex+1][0] - this->ky[yindex]*this->cu[tindex+0][0]);
        }
        else
            std::fill_n((rnumber*)(this->cvorticity+tindex), 6, 0.0);
    }
    );
    this->symmetrize(this->cvorticity, 3);
}

template <class rnumber>
void fluid_solver<rnumber>::compute_velocity(rnumber (*__restrict__ vorticity)[2])
{
    TIMEZONE("fluid_solver::compute_velocity");
    ptrdiff_t tindex;
    CLOOP_K2(
                this,
                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
        tindex = 3*cindex;
        if (k2 <= this->kM2 && k2 > 0)
        {
            this->cu[tindex+0][0] = -(this->ky[yindex]*vorticity[tindex+2][1] - this->kz[zindex]*vorticity[tindex+1][1]) / k2;
            this->cu[tindex+1][0] = -(this->kz[zindex]*vorticity[tindex+0][1] - this->kx[xindex]*vorticity[tindex+2][1]) / k2;
            this->cu[tindex+2][0] = -(this->kx[xindex]*vorticity[tindex+1][1] - this->ky[yindex]*vorticity[tindex+0][1]) / k2;
            this->cu[tindex+0][1] =  (this->ky[yindex]*vorticity[tindex+2][0] - this->kz[zindex]*vorticity[tindex+1][0]) / k2;
            this->cu[tindex+1][1] =  (this->kz[zindex]*vorticity[tindex+0][0] - this->kx[xindex]*vorticity[tindex+2][0]) / k2;
            this->cu[tindex+2][1] =  (this->kx[xindex]*vorticity[tindex+1][0] - this->ky[yindex]*vorticity[tindex+0][0]) / k2;
        }
        else
            std::fill_n((rnumber*)(this->cu+tindex), 6, 0.0);
    }
    );
    /*this->symmetrize(this->cu, 3);*/
}

template <class rnumber>
void fluid_solver<rnumber>::ift_velocity()
{
    TIMEZONE("fluid_solver::ift_velocity");
    fftw_interface<rnumber>::execute(*(this->c2r_velocity ));
}

template <class rnumber>
void fluid_solver<rnumber>::ift_vorticity()
{
    TIMEZONE("fluid_solver::ift_vorticity");
    std::fill_n(this->rvorticity, this->cd->local_size*2, 0.0);
    fftw_interface<rnumber>::execute(*(this->c2r_vorticity ));
}

template <class rnumber>
void fluid_solver<rnumber>::dft_velocity()
{
    TIMEZONE("fluid_solver::dft_velocity");
    fftw_interface<rnumber>::execute(*(this->r2c_velocity ));
}

template <class rnumber>
void fluid_solver<rnumber>::dft_vorticity()
{
    TIMEZONE("fluid_solver::dft_vorticity");
    std::fill_n((rnumber*)this->cvorticity, this->cd->local_size*2, 0.0);
    fftw_interface<rnumber>::execute(*(this->r2c_vorticity ));
}

template <class rnumber>
void fluid_solver<rnumber>::add_forcing(
        rnumber (*__restrict__ acc_field)[2], rnumber (*__restrict__ vort_field)[2], rnumber factor)
{
    TIMEZONE("fluid_solver::add_forcing");
    if (strcmp(this->forcing_type, "none") == 0)
        return;
    if (strcmp(this->forcing_type, "Kolmogorov") == 0)
    {
        ptrdiff_t cindex;
        if (this->cd->myrank == this->cd->rank[this->fmode])
        {
            cindex = ((this->fmode - this->cd->starts[0]) * this->cd->sizes[1])*this->cd->sizes[2]*3;
            acc_field[cindex+2][0] -= this->famplitude*factor/2;
        }
        if (this->cd->myrank == this->cd->rank[this->cd->sizes[0] - this->fmode])
        {
            cindex = ((this->cd->sizes[0] - this->fmode - this->cd->starts[0]) * this->cd->sizes[1])*this->cd->sizes[2]*3;
            acc_field[cindex+2][0] -= this->famplitude*factor/2;
        }
        return;
    }
    if (strcmp(this->forcing_type, "linear") == 0)
    {
        double knorm;
        CLOOP(
                    this,
                    [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
            knorm = sqrt(this->kx[xindex]*this->kx[xindex] +
                         this->ky[yindex]*this->ky[yindex] +
                         this->kz[zindex]*this->kz[zindex]);
            if ((this->fk0 <= knorm) &&
                    (this->fk1 >= knorm))
                for (int c=0; c<3; c++)
                    for (int i=0; i<2; i++)
                        acc_field[cindex*3+c][i] += this->famplitude*vort_field[cindex*3+c][i]*factor;
        }
        );
        return;
    }
}

template <class rnumber>
void fluid_solver<rnumber>::omega_nonlin(
        int src)
{
    TIMEZONE("fluid_solver::omega_nonlin");
    assert(src >= 0 && src < 3);
    this->compute_velocity(this->cv[src]);
    /* get fields from Fourier space to real space */
    {
        TIMEZONE("fluid_solver::omega_nonlin::fftw");
        fftw_interface<rnumber>::execute(*(this->c2r_velocity ));
        fftw_interface<rnumber>::execute(*(this->vc2r[src]));
    }
    /* compute cross product $u \times \omega$, and normalize */
    rnumber tmp[3][2];
    ptrdiff_t tindex;
    {
        TIMEZONE("fluid_solver::omega_nonlin::RLOOP");
        RLOOP (
                    this,
                    [&](ptrdiff_t rindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
            tindex = 3*rindex;
            for (int cc=0; cc<3; cc++)
                tmp[cc][0] = (this->ru[tindex+(cc+1)%3]*this->rv[src][tindex+(cc+2)%3] -
                        this->ru[tindex+(cc+2)%3]*this->rv[src][tindex+(cc+1)%3]);
            for (int cc=0; cc<3; cc++)
                this->ru[(3*rindex)+cc] = tmp[cc][0] / this->normalization_factor;
        }
        );
    }
    /* go back to Fourier space */
    this->clean_up_real_space(this->ru, 3);
    {
        TIMEZONE("fluid_solver::omega_nonlin::fftw-2");
        fftw_interface<rnumber>::execute(*(this->r2c_velocity ));
    }
    this->dealias(this->cu, 3);
    /* $\imath k \times Fourier(u \times \omega)$ */
    {
        TIMEZONE("fluid_solver::omega_nonlin::CLOOP");
        CLOOP(
                    this,
                    [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
            tindex = 3*cindex;
            {
                tmp[0][0] = -(this->ky[yindex]*this->cu[tindex+2][1] - this->kz[zindex]*this->cu[tindex+1][1]);
                tmp[1][0] = -(this->kz[zindex]*this->cu[tindex+0][1] - this->kx[xindex]*this->cu[tindex+2][1]);
                tmp[2][0] = -(this->kx[xindex]*this->cu[tindex+1][1] - this->ky[yindex]*this->cu[tindex+0][1]);
                tmp[0][1] =  (this->ky[yindex]*this->cu[tindex+2][0] - this->kz[zindex]*this->cu[tindex+1][0]);
                tmp[1][1] =  (this->kz[zindex]*this->cu[tindex+0][0] - this->kx[xindex]*this->cu[tindex+2][0]);
                tmp[2][1] =  (this->kx[xindex]*this->cu[tindex+1][0] - this->ky[yindex]*this->cu[tindex+0][0]);
            }
            for (int cc=0; cc<3; cc++) for (int i=0; i<2; i++)
                this->cu[tindex+cc][i] = tmp[cc][i];
        }
        );
    }
    {
        TIMEZONE("fluid_solver::omega_nonlin::add_forcing");
        this->add_forcing(this->cu, this->cv[src], 1.0);
    }
    {
        TIMEZONE("fluid_solver::omega_nonlin::force_divfree");
        this->force_divfree(this->cu);
    }
}

template <class rnumber>
void fluid_solver<rnumber>::step(double dt)
{
    TIMEZONE("fluid_solver::step");
    double factor0, factor1;
    std::fill_n((rnumber*)this->cv[1], this->cd->local_size*2, 0.0);
    this->omega_nonlin(0);
    CLOOP_K2(
                this,
                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
        if (k2 <= this->kM2)
        {
            factor0 = exp(-this->nu * k2 * dt);
            for (int cc=0; cc<3; cc++) for (int i=0; i<2; i++)
                this->cv[1][3*cindex+cc][i] = (this->cv[0][3*cindex+cc][i] +
                    dt*this->cu[3*cindex+cc][i])*factor0;
        }
    }
    );

    this->omega_nonlin(1);
    CLOOP_K2(
                this,
                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
        if (k2 <= this->kM2)
        {
            factor0 = exp(-this->nu * k2 * dt/2);
            factor1 = exp( this->nu * k2 * dt/2);
            for (int cc=0; cc<3; cc++) for (int i=0; i<2; i++)
                this->cv[2][3*cindex+cc][i] = (3*this->cv[0][3*cindex+cc][i]*factor0 +
                    (this->cv[1][3*cindex+cc][i] +
                    dt*this->cu[3*cindex+cc][i])*factor1)*0.25;
        }
    }
    );

    this->omega_nonlin(2);
    CLOOP_K2(
                this,
                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
        if (k2 <= this->kM2)
        {
            factor0 = exp(-this->nu * k2 * dt * 0.5);
            for (int cc=0; cc<3; cc++) for (int i=0; i<2; i++)
                this->cv[3][3*cindex+cc][i] = (this->cv[0][3*cindex+cc][i]*factor0 +
                    2*(this->cv[2][3*cindex+cc][i] +
                    dt*this->cu[3*cindex+cc][i]))*factor0/3;
        }
    }
    );

    this->force_divfree(this->cvorticity);
    this->symmetrize(this->cvorticity, 3);
    this->iteration++;
}

template <class rnumber>
int fluid_solver<rnumber>::read(char field, char representation)
{
    TIMEZONE("fluid_solver::read");
    char fname[512];
    int read_result;
    if (field == 'v')
    {
        if (representation == 'c')
        {
            this->fill_up_filename("cvorticity", fname);
            read_result = this->cd->read(fname, (void*)this->cvorticity);
            if (read_result != EXIT_SUCCESS)
                return read_result;
        }
        if (representation == 'r')
        {
            read_result = this->read_base("rvorticity", this->rvorticity);
            if (read_result != EXIT_SUCCESS)
                return read_result;
            else
                fftw_interface<rnumber>::execute(*(this->r2c_vorticity ));
        }
        this->low_pass_Fourier(this->cvorticity, 3, this->kM);
        this->force_divfree(this->cvorticity);
        this->symmetrize(this->cvorticity, 3);
        return EXIT_SUCCESS;
    }
    if ((field == 'u') && (representation == 'c'))
    {
        read_result = this->read_base("cvelocity", this->cvelocity);
        this->low_pass_Fourier(this->cvelocity, 3, this->kM);
        this->force_divfree(this->cvorticity);
        this->symmetrize(this->cvorticity, 3);
        return read_result;
    }
    if ((field == 'u') && (representation == 'r'))
        return this->read_base("rvelocity", this->rvelocity);
    return EXIT_FAILURE;
}

template <class rnumber>
int fluid_solver<rnumber>::write(char field, char representation)
{
    TIMEZONE("fluid_solver::write");
    char fname[512];
    if ((field == 'v') && (representation == 'c'))
    {
        this->fill_up_filename("cvorticity", fname);
        return this->cd->write(fname, (void*)this->cvorticity);
    }
    if ((field == 'v') && (representation == 'r'))
    {
        fftw_interface<rnumber>::execute(*(this->c2r_vorticity ));
        clip_zero_padding<rnumber>(this->rd, this->rvorticity, 3);
        this->fill_up_filename("rvorticity", fname);
        return this->rd->write(fname, this->rvorticity);
    }
    this->compute_velocity(this->cvorticity);
    if ((field == 'u') && (representation == 'c'))
    {
        this->fill_up_filename("cvelocity", fname);
        return this->cd->write(fname, this->cvelocity);
    }
    if ((field == 'u') && (representation == 'r'))
    {
        this->ift_velocity();
        clip_zero_padding<rnumber>(this->rd, this->rvelocity, 3);
        this->fill_up_filename("rvelocity", fname);
        return this->rd->write(fname, this->rvelocity);
    }
    return EXIT_FAILURE;
}

template <class rnumber>
int fluid_solver<rnumber>::write_rTrS2()
{
    TIMEZONE("fluid_solver::write_rTrS2");
    char fname[512];
    this->fill_up_filename("rTrS2", fname);
    typename fftw_interface<rnumber>::complex *ca;
    rnumber *ra;
    ca = fftw_interface<rnumber>::alloc_complex(this->cd->local_size*3);
    ra = (rnumber*)(ca);
    this->compute_velocity(this->cvorticity);
    this->compute_vector_gradient(ca, this->cvelocity);
    for (int cc=0; cc<3; cc++)
    {
        std::copy(
                    (rnumber*)(ca + cc*this->cd->local_size),
                    (rnumber*)(ca + (cc+1)*this->cd->local_size),
                    (rnumber*)this->cv[1]);
        fftw_interface<rnumber>::execute(*(this->vc2r[1]));
        std::copy(
                    this->rv[1],
                this->rv[1] + this->cd->local_size*2,
                ra + cc*this->cd->local_size*2);
    }
    /* velocity gradient is now stored, in real space, in ra */
    rnumber *dx_u, *dy_u, *dz_u;
    dx_u = ra;
    dy_u = ra + 2*this->cd->local_size;
    dz_u = ra + 4*this->cd->local_size;
    rnumber *trS2 = fftw_interface<rnumber>::alloc_real((this->cd->local_size/3)*2);
    double average_local = 0;
    RLOOP(
                this,
                [&](ptrdiff_t rindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
        rnumber AxxAxx;
        rnumber AyyAyy;
        rnumber AzzAzz;
        rnumber Sxy;
        rnumber Syz;
        rnumber Szx;
        ptrdiff_t tindex = 3*rindex;
        AxxAxx = dx_u[tindex+0]*dx_u[tindex+0];
        AyyAyy = dy_u[tindex+1]*dy_u[tindex+1];
        AzzAzz = dz_u[tindex+2]*dz_u[tindex+2];
        Sxy = dx_u[tindex+1]+dy_u[tindex+0];
        Syz = dy_u[tindex+2]+dz_u[tindex+1];
        Szx = dz_u[tindex+0]+dx_u[tindex+2];
        trS2[rindex] = (AxxAxx + AyyAyy + AzzAzz +
                        (Sxy*Sxy + Syz*Syz + Szx*Szx)/2);
        average_local += trS2[rindex];
    }
    );
    double average;
    MPI_Allreduce(
                &average_local,
                &average,
                1,
                MPI_DOUBLE, MPI_SUM, this->cd->comm);
    DEBUG_MSG("average TrS2 is %g\n", average);
    fftw_interface<rnumber>::free(ca);
    /* output goes here */
    int ntmp[3];
    ntmp[0] = this->rd->sizes[0];
    ntmp[1] = this->rd->sizes[1];
    ntmp[2] = this->rd->sizes[2];
    field_descriptor<rnumber> *scalar_descriptor = new field_descriptor<rnumber>(3, ntmp, mpi_real_type<rnumber>::real(), this->cd->comm);
    clip_zero_padding<rnumber>(scalar_descriptor, trS2, 1);
    int return_value = scalar_descriptor->write(fname, trS2);
    delete scalar_descriptor;
    fftw_interface<rnumber>::free(trS2);
    return return_value;
}

template <class rnumber>
int fluid_solver<rnumber>::write_renstrophy()
{
    TIMEZONE("fluid_solver::write_renstrophy");
    char fname[512];
    this->fill_up_filename("renstrophy", fname);
    rnumber *enstrophy = fftw_interface<rnumber>::alloc_real((this->cd->local_size/3)*2);
    this->ift_vorticity();
    double average_local = 0;
    RLOOP(
                this,
                [&](ptrdiff_t rindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
        ptrdiff_t tindex = 3*rindex;
        enstrophy[rindex] = (
                    this->rvorticity[tindex+0]*this->rvorticity[tindex+0] +
                this->rvorticity[tindex+1]*this->rvorticity[tindex+1] +
                this->rvorticity[tindex+2]*this->rvorticity[tindex+2]
                )/2;
        average_local += enstrophy[rindex];
    }
    );
    double average;
    MPI_Allreduce(
                &average_local,
                &average,
                1,
                MPI_DOUBLE, MPI_SUM, this->cd->comm);
    DEBUG_MSG("average enstrophy is %g\n", average);
    /* output goes here */
    int ntmp[3];
    ntmp[0] = this->rd->sizes[0];
    ntmp[1] = this->rd->sizes[1];
    ntmp[2] = this->rd->sizes[2];
    field_descriptor<rnumber> *scalar_descriptor = new field_descriptor<rnumber>(3, ntmp, mpi_real_type<rnumber>::real(), this->cd->comm);
    clip_zero_padding<rnumber>(scalar_descriptor, enstrophy, 1);
    int return_value = scalar_descriptor->write(fname, enstrophy);
    delete scalar_descriptor;
    fftw_interface<rnumber>::free(enstrophy);
    return return_value;
}

template <class rnumber>
void fluid_solver<rnumber>::compute_pressure(rnumber (*__restrict__ pressure)[2])
{
    TIMEZONE("fluid_solver::compute_pressure");
    /* assume velocity is already in real space representation */
    ptrdiff_t tindex;

    /* diagonal terms 11 22 33 */
    RLOOP (
                this,
                [&](ptrdiff_t rindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
        tindex = 3*rindex;
        for (int cc=0; cc<3; cc++)
            this->rv[1][tindex+cc] = this->ru[tindex+cc]*this->ru[tindex+cc];
    }
    );
    this->clean_up_real_space(this->rv[1], 3);
    {
        TIMEZONE("fftw_interface<rnumber>::execute");
        fftw_interface<rnumber>::execute(*(this->vr2c[1]));
    }
    this->dealias(this->cv[1], 3);
    CLOOP_K2(
                this,
                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
        if (k2 <= this->kM2 && k2 > 0)
        {
            tindex = 3*cindex;
            for (int i=0; i<2; i++)
            {
                pressure[cindex][i] = -(this->kx[xindex]*this->kx[xindex]*this->cv[1][tindex+0][i] +
                        this->ky[yindex]*this->ky[yindex]*this->cv[1][tindex+1][i] +
                        this->kz[zindex]*this->kz[zindex]*this->cv[1][tindex+2][i]);
            }
        }
        else
            std::fill_n((rnumber*)(pressure+cindex), 2, 0.0);
    }
    );
    /* off-diagonal terms 12 23 31 */
    RLOOP (
                this,
                [&](ptrdiff_t rindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
        tindex = 3*rindex;
        for (int cc=0; cc<3; cc++)
            this->rv[1][tindex+cc] = this->ru[tindex+cc]*this->ru[tindex+(cc+1)%3];
    }
    );
    this->clean_up_real_space(this->rv[1], 3);
    {
        TIMEZONE("fftw_interface<rnumber>::execute");
        fftw_interface<rnumber>::execute(*(this->vr2c[1]));
    }
    this->dealias(this->cv[1], 3);
    CLOOP_K2(
                this,
                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
        if (k2 <= this->kM2 && k2 > 0)
        {
            tindex = 3*cindex;
            for (int i=0; i<2; i++)
            {
                pressure[cindex][i] -= 2*(this->kx[xindex]*this->ky[yindex]*this->cv[1][tindex+0][i] +
                        this->ky[yindex]*this->kz[zindex]*this->cv[1][tindex+1][i] +
                        this->kz[zindex]*this->kx[xindex]*this->cv[1][tindex+2][i]);
                pressure[cindex][i] /= this->normalization_factor*k2;
            }
        }
    }
    );
}

template <class rnumber>
void fluid_solver<rnumber>::compute_gradient_statistics(
        rnumber (*__restrict__ vec)[2],
double *gradu_moments,
double *trS2QR_moments,
ptrdiff_t *gradu_hist,
ptrdiff_t *trS2QR_hist,
ptrdiff_t *QR2D_hist,
double trS2QR_max_estimates[],
double gradu_max_estimates[],
int nbins,
int QR2D_nbins)
{
    TIMEZONE("fluid_solver::compute_gradient_statistics");
    typename fftw_interface<rnumber>::complex *ca;
    rnumber *ra;
    ca = fftw_interface<rnumber>::alloc_complex(this->cd->local_size*3);
    ra = (rnumber*)(ca);
    this->compute_vector_gradient(ca, vec);
    for (int cc=0; cc<3; cc++)
    {
        std::copy(
                    (rnumber*)(ca + cc*this->cd->local_size),
                    (rnumber*)(ca + (cc+1)*this->cd->local_size),
                    (rnumber*)this->cv[1]);
        fftw_interface<rnumber>::execute(*(this->vc2r[1]));
        std::copy(
                    this->rv[1],
                this->rv[1] + this->cd->local_size*2,
                ra + cc*this->cd->local_size*2);
    }
    /* velocity gradient is now stored, in real space, in ra */
    std::fill_n(this->rv[1], 2*this->cd->local_size, 0.0);
    rnumber *dx_u, *dy_u, *dz_u;
    dx_u = ra;
    dy_u = ra + 2*this->cd->local_size;
    dz_u = ra + 4*this->cd->local_size;
    double binsize[2];
    double tmp_max_estimate[3];
    tmp_max_estimate[0] = trS2QR_max_estimates[0];
    tmp_max_estimate[1] = trS2QR_max_estimates[1];
    tmp_max_estimate[2] = trS2QR_max_estimates[2];
    binsize[0] = 2*tmp_max_estimate[2] / QR2D_nbins;
    binsize[1] = 2*tmp_max_estimate[1] / QR2D_nbins;
    ptrdiff_t *local_hist = new ptrdiff_t[QR2D_nbins*QR2D_nbins];
    std::fill_n(local_hist, QR2D_nbins*QR2D_nbins, 0);
    RLOOP(
                this,
                [&](ptrdiff_t rindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
        rnumber AxxAxx;
        rnumber AyyAyy;
        rnumber AzzAzz;
        rnumber AxyAyx;
        rnumber AyzAzy;
        rnumber AzxAxz;
        rnumber Sxy;
        rnumber Syz;
        rnumber Szx;
        ptrdiff_t tindex = 3*rindex;
        AxxAxx = dx_u[tindex+0]*dx_u[tindex+0];
        AyyAyy = dy_u[tindex+1]*dy_u[tindex+1];
        AzzAzz = dz_u[tindex+2]*dz_u[tindex+2];
        AxyAyx = dx_u[tindex+1]*dy_u[tindex+0];
        AyzAzy = dy_u[tindex+2]*dz_u[tindex+1];
        AzxAxz = dz_u[tindex+0]*dx_u[tindex+2];
        this->rv[1][tindex+1] = - (AxxAxx + AyyAyy + AzzAzz)/2 - AxyAyx - AyzAzy - AzxAxz;
        this->rv[1][tindex+2] = - (dx_u[tindex+0]*(AxxAxx/3 + AxyAyx + AzxAxz) +
                dy_u[tindex+1]*(AyyAyy/3 + AxyAyx + AyzAzy) +
                dz_u[tindex+2]*(AzzAzz/3 + AzxAxz + AyzAzy) +
                dx_u[tindex+1]*dy_u[tindex+2]*dz_u[tindex+0] +
                dx_u[tindex+2]*dy_u[tindex+0]*dz_u[tindex+1]);
        int bin0 = int(floor((this->rv[1][tindex+2] + tmp_max_estimate[2]) / binsize[0]));
        int bin1 = int(floor((this->rv[1][tindex+1] + tmp_max_estimate[1]) / binsize[1]));
        if ((bin0 >= 0 && bin0 < QR2D_nbins) &&
                (bin1 >= 0 && bin1 < QR2D_nbins))
            local_hist[bin1*QR2D_nbins + bin0]++;
        Sxy = dx_u[tindex+1]+dy_u[tindex+0];
        Syz = dy_u[tindex+2]+dz_u[tindex+1];
        Szx = dz_u[tindex+0]+dx_u[tindex+2];
        this->rv[1][tindex] = (AxxAxx + AyyAyy + AzzAzz +
                               (Sxy*Sxy + Syz*Syz + Szx*Szx)/2);
    }
    );
    MPI_Allreduce(
                local_hist,
                QR2D_hist,
                QR2D_nbins * QR2D_nbins,
                MPI_INT64_T, MPI_SUM, this->cd->comm);
    delete[] local_hist;
    this->compute_rspace_stats3(
                this->rv[1],
            trS2QR_moments,
            trS2QR_hist,
            tmp_max_estimate,
            nbins);
    double *tmp_moments = new double[10*3];
    ptrdiff_t *tmp_hist = new ptrdiff_t[nbins*3];
    for (int cc=0; cc<3; cc++)
    {
        tmp_max_estimate[0] = gradu_max_estimates[cc*3 + 0];
        tmp_max_estimate[1] = gradu_max_estimates[cc*3 + 1];
        tmp_max_estimate[2] = gradu_max_estimates[cc*3 + 2];
        this->compute_rspace_stats3(
                    dx_u + cc*2*this->cd->local_size,
                    tmp_moments,
                    tmp_hist,
                    tmp_max_estimate,
                    nbins);
        for (int n = 0; n < 10; n++)
            for (int i = 0; i < 3 ; i++)
            {
                gradu_moments[(n*3 + cc)*3 + i] = tmp_moments[n*3 + i];
            }
        for (int n = 0; n < nbins; n++)
            for (int i = 0; i < 3; i++)
            {
                gradu_hist[(n*3 + cc)*3 + i] = tmp_hist[n*3 + i];
            }
    }
    delete[] tmp_moments;
    delete[] tmp_hist;
    fftw_interface<rnumber>::free(ca);
}

template <class rnumber>
void fluid_solver<rnumber>::compute_Lagrangian_acceleration(rnumber (*acceleration)[2])
{
    TIMEZONE("fluid_solver::compute_Lagrangian_acceleration");
    ptrdiff_t tindex;
    typename fftw_interface<rnumber>::complex *pressure;
    pressure = fftw_interface<rnumber>::alloc_complex(this->cd->local_size/3);
    this->compute_velocity(this->cvorticity);
    this->ift_velocity();
    this->compute_pressure(pressure);
    this->compute_velocity(this->cvorticity);
    std::fill_n((rnumber*)this->cv[1], 2*this->cd->local_size, 0.0);
    CLOOP_K2(
                this,
                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
        if (k2 <= this->kM2)
        {
            tindex = 3*cindex;
            for (int cc=0; cc<3; cc++)
                for (int i=0; i<2; i++)
                    this->cv[1][tindex+cc][i] = - this->nu*k2*this->cu[tindex+cc][i];
            if (strcmp(this->forcing_type, "linear") == 0)
            {
                double knorm = sqrt(k2);
                if ((this->fk0 <= knorm) &&
                        (this->fk1 >= knorm))
                    for (int c=0; c<3; c++)
                        for (int i=0; i<2; i++)
                            this->cv[1][tindex+c][i] += this->famplitude*this->cu[tindex+c][i];
            }
            this->cv[1][tindex+0][0] += this->kx[xindex]*pressure[cindex][1];
            this->cv[1][tindex+1][0] += this->ky[yindex]*pressure[cindex][1];
            this->cv[1][tindex+2][0] += this->kz[zindex]*pressure[cindex][1];
            this->cv[1][tindex+0][1] -= this->kx[xindex]*pressure[cindex][0];
            this->cv[1][tindex+1][1] -= this->ky[yindex]*pressure[cindex][0];
            this->cv[1][tindex+2][1] -= this->kz[zindex]*pressure[cindex][0];
        }
    }
    );
    std::copy(
                (rnumber*)this->cv[1],
            (rnumber*)(this->cv[1] + this->cd->local_size),
            (rnumber*)acceleration);
    fftw_interface<rnumber>::free(pressure);
}

template <class rnumber>
void fluid_solver<rnumber>::compute_Eulerian_acceleration(rnumber (*__restrict__ acceleration)[2])
{
    TIMEZONE("fluid_solver::compute_Eulerian_acceleration");
    std::fill_n((rnumber*)(acceleration), 2*this->cd->local_size, 0.0);
    ptrdiff_t tindex;
    this->compute_velocity(this->cvorticity);
    /* put in linear terms */
    CLOOP_K2(
                this,
                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
        if (k2 <= this->kM2)
        {
            tindex = 3*cindex;
            for (int cc=0; cc<3; cc++)
                for (int i=0; i<2; i++)
                    acceleration[tindex+cc][i] = - this->nu*k2*this->cu[tindex+cc][i];
            if (strcmp(this->forcing_type, "linear") == 0)
            {
                double knorm = sqrt(k2);
                if ((this->fk0 <= knorm) &&
                        (this->fk1 >= knorm))
                {
                    for (int c=0; c<3; c++)
                        for (int i=0; i<2; i++)
                            acceleration[tindex+c][i] += this->famplitude*this->cu[tindex+c][i];
                }
            }
        }
    }
    );
    this->ift_velocity();
    /* compute uu */
    /* 11 22 33 */
    RLOOP (
                this,
                [&](ptrdiff_t rindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
        tindex = 3*rindex;
        for (int cc=0; cc<3; cc++)
            this->rv[1][tindex+cc] = this->ru[tindex+cc]*this->ru[tindex+cc] / this->normalization_factor;
    }
    );
    this->clean_up_real_space(this->rv[1], 3);
    fftw_interface<rnumber>::execute(*(this->vr2c[1]));
    this->dealias(this->cv[1], 3);
    CLOOP_K2(
                this,
                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
        if (k2 <= this->kM2)
        {
            tindex = 3*cindex;
            acceleration[tindex+0][0] +=
                    this->kx[xindex]*this->cv[1][tindex+0][1];
            acceleration[tindex+0][1] +=
                    -this->kx[xindex]*this->cv[1][tindex+0][0];
            acceleration[tindex+1][0] +=
                    this->ky[yindex]*this->cv[1][tindex+1][1];
            acceleration[tindex+1][1] +=
                    -this->ky[yindex]*this->cv[1][tindex+1][0];
            acceleration[tindex+2][0] +=
                    this->kz[zindex]*this->cv[1][tindex+2][1];
            acceleration[tindex+2][1] +=
                    -this->kz[zindex]*this->cv[1][tindex+2][0];
        }
    }
    );
    /* 12 23 31 */
    RLOOP (
                this,
                [&](ptrdiff_t rindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
        tindex = 3*rindex;
        for (int cc=0; cc<3; cc++)
            this->rv[1][tindex+cc] = this->ru[tindex+cc]*this->ru[tindex+(cc+1)%3] / this->normalization_factor;
    }
    );
    this->clean_up_real_space(this->rv[1], 3);
    fftw_interface<rnumber>::execute(*(this->vr2c[1]));
    this->dealias(this->cv[1], 3);
    CLOOP_K2(
                this,
                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
        if (k2 <= this->kM2)
        {
            tindex = 3*cindex;
            acceleration[tindex+0][0] +=
                    (this->ky[yindex]*this->cv[1][tindex+0][1] +
                    this->kz[zindex]*this->cv[1][tindex+2][1]);
            acceleration[tindex+0][1] +=
                    - (this->ky[yindex]*this->cv[1][tindex+0][0] +
                    this->kz[zindex]*this->cv[1][tindex+2][0]);
            acceleration[tindex+1][0] +=
                    (this->kz[zindex]*this->cv[1][tindex+1][1] +
                    this->kx[xindex]*this->cv[1][tindex+0][1]);
            acceleration[tindex+1][1] +=
                    - (this->kz[zindex]*this->cv[1][tindex+1][0] +
                    this->kx[xindex]*this->cv[1][tindex+0][0]);
            acceleration[tindex+2][0] +=
                    (this->kx[xindex]*this->cv[1][tindex+2][1] +
                    this->ky[yindex]*this->cv[1][tindex+1][1]);
            acceleration[tindex+2][1] +=
                    - (this->kx[xindex]*this->cv[1][tindex+2][0] +
                    this->ky[yindex]*this->cv[1][tindex+1][0]);
        }
    }
    );
    if (this->cd->myrank == this->cd->rank[0])
        std::fill_n((rnumber*)(acceleration), 6, 0.0);
    this->force_divfree(acceleration);
}

template <class rnumber>
void fluid_solver<rnumber>::compute_Lagrangian_acceleration(rnumber *__restrict__ acceleration)
{
    TIMEZONE("fluid_solver::compute_Lagrangian_acceleration");
    this->compute_Lagrangian_acceleration((typename fftw_interface<rnumber>::complex*)acceleration);
    fftw_interface<rnumber>::execute(*(this->vc2r[1]));
    std::copy(
                this->rv[1],
            this->rv[1] + 2*this->cd->local_size,
            acceleration);
}

template <class rnumber>
int fluid_solver<rnumber>::write_rpressure()
{
    TIMEZONE("fluid_solver::write_rpressure");
    char fname[512];
    typename fftw_interface<rnumber>::complex *pressure;
    pressure = fftw_interface<rnumber>::alloc_complex(this->cd->local_size/3);
    this->compute_velocity(this->cvorticity);
    this->ift_velocity();
    this->compute_pressure(pressure);
    this->fill_up_filename("rpressure", fname);
    rnumber *rpressure = fftw_interface<rnumber>::alloc_real((this->cd->local_size/3)*2);
    typename fftw_interface<rnumber>::plan c2r;
    c2r = fftw_interface<rnumber>::mpi_plan_dft_c2r_3d(
                this->rd->sizes[0], this->rd->sizes[1], this->rd->sizes[2],
            pressure, rpressure, this->cd->comm,
            this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_IN);
    fftw_interface<rnumber>::execute(c2r);
    /* output goes here */
    int ntmp[3];
    ntmp[0] = this->rd->sizes[0];
    ntmp[1] = this->rd->sizes[1];
    ntmp[2] = this->rd->sizes[2];
    field_descriptor<rnumber> *scalar_descriptor = new field_descriptor<rnumber>(3, ntmp, mpi_real_type<rnumber>::real(), this->cd->comm);
    clip_zero_padding<rnumber>(scalar_descriptor, rpressure, 1);
    int return_value = scalar_descriptor->write(fname, rpressure);
    delete scalar_descriptor;
    fftw_interface<rnumber>::destroy_plan(c2r);
    fftw_interface<rnumber>::free(pressure);
    fftw_interface<rnumber>::free(rpressure);
    return return_value;
}

/*****************************************************************************/




/*****************************************************************************/
/* finally, force generation of code for single precision                    */
template class fluid_solver<float>;
template class fluid_solver<double>;
/*****************************************************************************/

