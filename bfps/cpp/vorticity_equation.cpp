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
#include "fftw_tools.hpp"
#include "vorticity_equation.hpp"
#include "scope_timer.hpp"



template <class rnumber,
          field_backend be>
void vorticity_equation<rnumber, be>::impose_zero_modes()
{
    this->u->impose_zero_mode();
    this->v[0]->impose_zero_mode();
    this->v[1]->impose_zero_mode();
    this->v[2]->impose_zero_mode();
}

template <class rnumber,
          field_backend be>
vorticity_equation<rnumber, be>::vorticity_equation(
        const char *NAME,
        int nx,
        int ny,
        int nz,
        double DKX,
        double DKY,
        double DKZ,
        unsigned FFTW_PLAN_RIGOR)
{
    /* initialize name and basic stuff */
    strncpy(this->name, NAME, 256);
    this->name[255] = '\0';
    this->iteration = 0;

    /* initialize field descriptors */
    int ntmp[4];
    ntmp[0] = nz;
    ntmp[1] = ny;
    ntmp[2] = nx;
    ntmp[3] = 3;
    this->rd = new field_descriptor<rnumber>(
                4, ntmp, mpi_real_type<rnumber>::real(), MPI_COMM_WORLD);
    ntmp[0] = ny;
    ntmp[1] = nz;
    ntmp[2] = nx/2 + 1;
    ntmp[3] = 3;
    this->cd = new field_descriptor<rnumber>(
                4, ntmp, mpi_real_type<rnumber>::complex(), this->rd->comm);

    /* initialize fields */
    this->cvorticity = new field<rnumber, be, THREE>(
            nx, ny, nz, MPI_COMM_WORLD, FFTW_PLAN_RIGOR);
    this->rvorticity = new field<rnumber, be, THREE>(
            nx, ny, nz, MPI_COMM_WORLD, FFTW_PLAN_RIGOR);
    this->v[1] = new field<rnumber, be, THREE>(
            nx, ny, nz, MPI_COMM_WORLD, FFTW_PLAN_RIGOR);
    this->v[2] = new field<rnumber, be, THREE>(
            nx, ny, nz, MPI_COMM_WORLD, FFTW_PLAN_RIGOR);
    this->v[0] = this->cvorticity;
    this->v[3] = this->cvorticity;

    this->cvelocity = new field<rnumber, be, THREE>(
            nx, ny, nz, MPI_COMM_WORLD, FFTW_PLAN_RIGOR);
    this->rvelocity = new field<rnumber, be, THREE>(
            nx, ny, nz, MPI_COMM_WORLD, FFTW_PLAN_RIGOR);
    this->u = this->cvelocity;

    /* initialize kspace */
    this->kk = new kspace<be, SMOOTH>(
            this->cvorticity->clayout, DKX, DKY, DKZ);

    /* ``physical'' parameters etc, initialized here just in case */

    this->nu = 0.1;
    this->fmode = 1;
    this->famplitude = 1.0;
    this->fk0  = 2.0;
    this->fk1 = 4.0;
}

template <class rnumber,
          field_backend be>
vorticity_equation<rnumber, be>::~vorticity_equation()
{
    delete this->kk;
    delete this->cvorticity;
    delete this->rvorticity;
    delete this->v[1];
    delete this->v[2];
    delete this->cvelocity;
    delete this->rvelocity;
}

template <class rnumber,
          field_backend be>
void vorticity_equation<rnumber, be>::compute_vorticity()
{
    this->kk->CLOOP_K2(
                [&](ptrdiff_t cindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex,
                    double k2){
        if (k2 <= this->kk->kM2)
        {
            ptrdiff_t tindex = 3*cindex;
            this->cvorticity->get_cdata()[tindex+0][0] = -(this->kk->ky[yindex]*this->u->get_cdata()[tindex+2][1] - this->kk->kz[zindex]*this->u->get_cdata()[tindex+1][1]);
            this->cvorticity->get_cdata()[tindex+1][0] = -(this->kk->kz[zindex]*this->u->get_cdata()[tindex+0][1] - this->kk->kx[xindex]*this->u->get_cdata()[tindex+2][1]);
            this->cvorticity->get_cdata()[tindex+2][0] = -(this->kk->kx[xindex]*this->u->get_cdata()[tindex+1][1] - this->kk->ky[yindex]*this->u->get_cdata()[tindex+0][1]);
            this->cvorticity->get_cdata()[tindex+0][1] =  (this->kk->ky[yindex]*this->u->get_cdata()[tindex+2][0] - this->kk->kz[zindex]*this->u->get_cdata()[tindex+1][0]);
            this->cvorticity->get_cdata()[tindex+1][1] =  (this->kk->kz[zindex]*this->u->get_cdata()[tindex+0][0] - this->kk->kx[xindex]*this->u->get_cdata()[tindex+2][0]);
            this->cvorticity->get_cdata()[tindex+2][1] =  (this->kk->kx[xindex]*this->u->get_cdata()[tindex+1][0] - this->kk->ky[yindex]*this->u->get_cdata()[tindex+0][0]);
        }
        else
            std::fill_n((rnumber*)(this->cvorticity->get_cdata()+3*cindex), 6, 0.0);
    }
    );
    this->cvorticity->symmetrize();
}

template <class rnumber,
          field_backend be>
void vorticity_equation<rnumber, be>::compute_velocity(field<rnumber, be, THREE> *vorticity)
{
    this->u->real_space_representation = false;
    this->kk->CLOOP_K2(
                [&](ptrdiff_t cindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex,
                    double k2){
        if (k2 <= this->kk->kM2 && k2 > 0)
        {
            ptrdiff_t tindex = 3*cindex;
            this->u->get_cdata()[tindex+0][0] = -(this->kk->ky[yindex]*vorticity->get_cdata()[tindex+2][1] - this->kk->kz[zindex]*vorticity->get_cdata()[tindex+1][1]) / k2;
            this->u->get_cdata()[tindex+1][0] = -(this->kk->kz[zindex]*vorticity->get_cdata()[tindex+0][1] - this->kk->kx[xindex]*vorticity->get_cdata()[tindex+2][1]) / k2;
            this->u->get_cdata()[tindex+2][0] = -(this->kk->kx[xindex]*vorticity->get_cdata()[tindex+1][1] - this->kk->ky[yindex]*vorticity->get_cdata()[tindex+0][1]) / k2;
            this->u->get_cdata()[tindex+0][1] =  (this->kk->ky[yindex]*vorticity->get_cdata()[tindex+2][0] - this->kk->kz[zindex]*vorticity->get_cdata()[tindex+1][0]) / k2;
            this->u->get_cdata()[tindex+1][1] =  (this->kk->kz[zindex]*vorticity->get_cdata()[tindex+0][0] - this->kk->kx[xindex]*vorticity->get_cdata()[tindex+2][0]) / k2;
            this->u->get_cdata()[tindex+2][1] =  (this->kk->kx[xindex]*vorticity->get_cdata()[tindex+1][0] - this->kk->ky[yindex]*vorticity->get_cdata()[tindex+0][0]) / k2;
        }
        else
            std::fill_n((rnumber*)(this->u->get_cdata()+3*cindex), 6, 0.0);
    }
    );
    this->u->symmetrize();
}

template <class rnumber,
          field_backend be>
void vorticity_equation<rnumber, be>::add_forcing(
        field<rnumber, be, THREE> *dst,
        field<rnumber, be, THREE> *vort_field,
        rnumber factor)
{
    if (strcmp(this->forcing_type, "none") == 0)
        return;
    if (strcmp(this->forcing_type, "Kolmogorov") == 0)
    {
        ptrdiff_t cindex;
        if (this->cd->myrank == this->cd->rank[this->fmode])
        {
            cindex = ((this->fmode - this->cd->starts[0]) * this->cd->sizes[1])*this->cd->sizes[2]*3;
            dst->get_cdata()[cindex+2][0] -= this->famplitude*factor/2;
        }
        if (this->cd->myrank == this->cd->rank[this->cd->sizes[0] - this->fmode])
        {
            cindex = ((this->cd->sizes[0] - this->fmode - this->cd->starts[0]) * this->cd->sizes[1])*this->cd->sizes[2]*3;
            dst->get_cdata()[cindex+2][0] -= this->famplitude*factor/2;
        }
        return;
    }
    if (strcmp(this->forcing_type, "linear") == 0)
    {
        this->kk->CLOOP(
                    [&](ptrdiff_t cindex,
                        ptrdiff_t xindex,
                        ptrdiff_t yindex,
                        ptrdiff_t zindex){
            double knorm = sqrt(this->kk->kx[xindex]*this->kk->kx[xindex] +
                                this->kk->ky[yindex]*this->kk->ky[yindex] +
                                this->kk->kz[zindex]*this->kk->kz[zindex]);
            if ((this->fk0 <= knorm) &&
                    (this->fk1 >= knorm))
                for (int c=0; c<3; c++)
                    for (int i=0; i<2; i++)
                        dst->get_cdata()[cindex*3+c][i] += \
                            this->famplitude*vort_field->get_cdata()[cindex*3+c][i]*factor;
        }
        );
        return;
    }
}

template <class rnumber,
          field_backend be>
void vorticity_equation<rnumber, be>::omega_nonlin(
        int src)
{
    DEBUG_MSG("vorticity_equation::omega_nonlin(%d)\n", src);
    assert(src >= 0 && src < 3);
    this->compute_velocity(this->v[src]);
    /* get fields from Fourier space to real space */
    this->u->ift();
    this->rvorticity->real_space_representation = false;
    *this->rvorticity = this->v[src]->get_cdata();
    this->rvorticity->ift();
    /* compute cross product $u \times \omega$, and normalize */
    rnumber tmp[3][2];
    this->u->RLOOP(
                [&](ptrdiff_t rindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex){
        ptrdiff_t tindex = 3*rindex;
        for (int cc=0; cc<3; cc++)
            tmp[cc][0] = (this->u->get_rdata()[tindex+(cc+1)%3]*this->rvorticity->get_rdata()[tindex+(cc+2)%3] -
                          this->u->get_rdata()[tindex+(cc+2)%3]*this->rvorticity->get_rdata()[tindex+(cc+1)%3]);
        for (int cc=0; cc<3; cc++)
            this->u->get_rdata()[(3*rindex)+cc] = tmp[cc][0] / this->u->npoints;
    }
    );
    /* go back to Fourier space */
    //this->clean_up_real_space(this->ru, 3);
    this->u->dft();
    this->kk->template dealias<rnumber, THREE>(this->u->get_cdata());
    /* $\imath k \times Fourier(u \times \omega)$ */
    this->kk->CLOOP(
                [&](ptrdiff_t cindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex){
        ptrdiff_t tindex = 3*cindex;
        {
            tmp[0][0] = -(this->kk->ky[yindex]*this->u->get_cdata()[tindex+2][1] - this->kk->kz[zindex]*this->u->get_cdata()[tindex+1][1]);
            tmp[1][0] = -(this->kk->kz[zindex]*this->u->get_cdata()[tindex+0][1] - this->kk->kx[xindex]*this->u->get_cdata()[tindex+2][1]);
            tmp[2][0] = -(this->kk->kx[xindex]*this->u->get_cdata()[tindex+1][1] - this->kk->ky[yindex]*this->u->get_cdata()[tindex+0][1]);
            tmp[0][1] =  (this->kk->ky[yindex]*this->u->get_cdata()[tindex+2][0] - this->kk->kz[zindex]*this->u->get_cdata()[tindex+1][0]);
            tmp[1][1] =  (this->kk->kz[zindex]*this->u->get_cdata()[tindex+0][0] - this->kk->kx[xindex]*this->u->get_cdata()[tindex+2][0]);
            tmp[2][1] =  (this->kk->kx[xindex]*this->u->get_cdata()[tindex+1][0] - this->kk->ky[yindex]*this->u->get_cdata()[tindex+0][0]);
        }
        for (int cc=0; cc<3; cc++) for (int i=0; i<2; i++)
            this->u->get_cdata()[tindex+cc][i] = tmp[cc][i];
    }
    );
    this->add_forcing(this->u, this->v[src], 1.0);
    this->kk->template force_divfree<rnumber>(this->u->get_cdata());
}

template <class rnumber,
          field_backend be>
void vorticity_equation<rnumber, be>::step(double dt)
{
    DEBUG_MSG("vorticity_equation::step\n");
    TIMEZONE("vorticity_equation::step");
    double factor0, factor1;
    *this->v[1] = 0.0;
    this->omega_nonlin(0);
    this->kk->CLOOP_K2(
                [&](ptrdiff_t cindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex,
                    double k2){
        if (k2 <= this->kk->kM2)
        {
            factor0 = exp(-this->nu * k2 * dt);
            for (int cc=0; cc<3; cc++) for (int i=0; i<2; i++)
                this->v[1]->get_cdata()[3*cindex+cc][i] = (
                        this->v[0]->get_cdata()[3*cindex+cc][i] +
                        dt*this->u->get_cdata()[3*cindex+cc][i])*factor0;
        }
    }
    );

    this->omega_nonlin(1);
    this->kk->CLOOP_K2(
                [&](ptrdiff_t cindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex,
                    double k2){
        if (k2 <= this->kk->kM2)
        {
            factor0 = exp(-this->nu * k2 * dt/2);
            factor1 = exp( this->nu * k2 * dt/2);
            for (int cc=0; cc<3; cc++) for (int i=0; i<2; i++)
                this->v[2]->get_cdata()[3*cindex+cc][i] = (
                        3*this->v[0]->get_cdata()[3*cindex+cc][i]*factor0 +
                        (this->v[1]->get_cdata()[3*cindex+cc][i] +
                         dt*this->u->get_cdata()[3*cindex+cc][i])*factor1)*0.25;
        }
    }
    );

    this->omega_nonlin(2);
    this->kk->CLOOP_K2(
                [&](ptrdiff_t cindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex,
                    double k2){
        if (k2 <= this->kk->kM2)
        {
            factor0 = exp(-this->nu * k2 * dt * 0.5);
            for (int cc=0; cc<3; cc++) for (int i=0; i<2; i++)
                this->v[3]->get_cdata()[3*cindex+cc][i] = (
                        this->v[0]->get_cdata()[3*cindex+cc][i]*factor0 +
                        2*(this->v[2]->get_cdata()[3*cindex+cc][i] +
                           dt*this->u->get_cdata()[3*cindex+cc][i]))*factor0/3;
        }
    }
    );

    this->kk->template force_divfree<rnumber>(this->cvorticity->get_cdata());
    this->cvorticity->symmetrize();
    this->iteration++;
}

template <class rnumber,
          field_backend be>
int vorticity_equation<rnumber, be>::read(char field, char representation)
{
    char fname[512];
    int read_result;
    if (field == 'v')
    {
        if (representation == 'c')
        {
            this->fill_up_filename("cvorticity", fname);
            this->cvorticity->real_space_representation = false;
            read_result = this->cd->read(fname, this->cvorticity->get_cdata());
            if (read_result != EXIT_SUCCESS)
                return read_result;
        }
        if (representation == 'r')
        {
            this->fill_up_filename("rvorticity", fname);
            this->cvorticity->real_space_representation = true;
            read_result = this->rd->read(fname, this->cvorticity->get_rdata());
            if (read_result != EXIT_SUCCESS)
                return read_result;
            else
                this->cvorticity->dft();
        }
        this->kk->template low_pass<rnumber, THREE>(this->cvorticity->get_rdata(), this->kk->kM);
        this->kk->template force_divfree<rnumber>(this->cvorticity->get_cdata());
        this->cvorticity->symmetrize();
        return EXIT_SUCCESS;
    }
    if ((field == 'u') && (representation == 'c'))
    {
        this->fill_up_filename("cvelocity", fname);
        read_result = this->cd->read(fname, this->cvelocity->get_cdata());
        this->kk->template low_pass<rnumber, THREE>(this->cvelocity->get_rdata(), this->kk->kM);
        this->compute_vorticity();
        this->kk->template force_divfree<rnumber>(this->cvorticity->get_cdata());
        this->cvorticity->symmetrize();
        return read_result;
    }
    if ((field == 'u') && (representation == 'r'))
    {
        this->fill_up_filename("rvelocity", fname);
        this->u->real_space_representation = true;
        return this->rd->read(fname, this->u->get_rdata());
    }
    return EXIT_FAILURE;
}

template <class rnumber,
          field_backend be>
int vorticity_equation<rnumber, be>::write(char field, char representation)
{
    char fname[512];
    if ((field == 'v') && (representation == 'c'))
    {
        this->fill_up_filename("cvorticity", fname);
        return this->cd->write(fname, (void*)this->cvorticity);
    }
    if ((field == 'v') && (representation == 'r'))
    {
        *this->rvorticity = this->cvorticity->get_cdata();
        this->rvorticity->ift();
        clip_zero_padding<rnumber>(this->rd, this->rvorticity->get_rdata(), 3);
        this->fill_up_filename("rvorticity", fname);
        return this->rd->write(fname, this->rvorticity->get_rdata());
    }
    this->compute_velocity(this->cvorticity);
    if ((field == 'u') && (representation == 'c'))
    {
        this->fill_up_filename("cvelocity", fname);
        return this->cd->write(fname, this->cvelocity->get_cdata());
    }
    if ((field == 'u') && (representation == 'r'))
    {
        *this->rvelocity = this->cvelocity->get_cdata();
        this->rvelocity->ift();
        clip_zero_padding<rnumber>(this->rd, this->rvelocity->get_rdata(), 3);
        this->fill_up_filename("rvelocity", fname);
        return this->rd->write(fname, this->rvelocity->get_rdata());
    }
    return EXIT_FAILURE;
}

//template <class rnumber,
//          field_backend be>
//int vorticity_equation<rnumber, be>::write_rTrS2()
//{
//    char fname[512];
//    this->fill_up_filename("rTrS2", fname);
//    typename fftw_interface<rnumber>::complex *ca;
//    rnumber *ra;
//    ca = fftw_interface<rnumber>::alloc_complex(this->cd->local_size*3);
//    ra = (rnumber*)(ca);
//    this->compute_velocity(this->cvorticity);
//    this->compute_vector_gradient(ca, this->cvelocity);
//    for (int cc=0; cc<3; cc++)
//    {
//        std::copy(
//                    (rnumber*)(ca + cc*this->cd->local_size),
//                    (rnumber*)(ca + (cc+1)*this->cd->local_size),
//                    (rnumber*)this->cv[1]);
//        fftw_interface<rnumber>::execute(*(this->vc2r[1]));
//        std::copy(
//                    this->rv[1],
//                this->rv[1] + this->cd->local_size*2,
//                ra + cc*this->cd->local_size*2);
//    }
//    /* velocity gradient is now stored, in real space, in ra */
//    rnumber *dx_u, *dy_u, *dz_u;
//    dx_u = ra;
//    dy_u = ra + 2*this->cd->local_size;
//    dz_u = ra + 4*this->cd->local_size;
//    rnumber *trS2 = fftw_interface<rnumber>::alloc_real((this->cd->local_size/3)*2);
//    double average_local = 0;
//    RLOOP(
//                this,
//                [&](ptrdiff_t rindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
//        rnumber AxxAxx;
//        rnumber AyyAyy;
//        rnumber AzzAzz;
//        rnumber Sxy;
//        rnumber Syz;
//        rnumber Szx;
//        ptrdiff_t tindex = 3*rindex;
//        AxxAxx = dx_u[tindex+0]*dx_u[tindex+0];
//        AyyAyy = dy_u[tindex+1]*dy_u[tindex+1];
//        AzzAzz = dz_u[tindex+2]*dz_u[tindex+2];
//        Sxy = dx_u[tindex+1]+dy_u[tindex+0];
//        Syz = dy_u[tindex+2]+dz_u[tindex+1];
//        Szx = dz_u[tindex+0]+dx_u[tindex+2];
//        trS2[rindex] = (AxxAxx + AyyAyy + AzzAzz +
//                        (Sxy*Sxy + Syz*Syz + Szx*Szx)/2);
//        average_local += trS2[rindex];
//    }
//    );
//    double average;
//    MPI_Allreduce(
//                &average_local,
//                &average,
//                1,
//                MPI_DOUBLE, MPI_SUM, this->cd->comm);
//    DEBUG_MSG("average TrS2 is %g\n", average);
//    fftw_interface<rnumber>::free(ca);
//    /* output goes here */
//    int ntmp[3];
//    ntmp[0] = this->rd->sizes[0];
//    ntmp[1] = this->rd->sizes[1];
//    ntmp[2] = this->rd->sizes[2];
//    field_descriptor<rnumber> *scalar_descriptor = new field_descriptor<rnumber>(3, ntmp, mpi_real_type<rnumber>::real(), this->cd->comm);
//    clip_zero_padding<rnumber>(scalar_descriptor, trS2, 1);
//    int return_value = scalar_descriptor->write(fname, trS2);
//    delete scalar_descriptor;
//    fftw_interface<rnumber>::free(trS2);
//    return return_value;
//}
//
//template <class rnumber,
//          field_backend be>
//int vorticity_equation<rnumber, be>::write_renstrophy()
//{
//    char fname[512];
//    this->fill_up_filename("renstrophy", fname);
//    rnumber *enstrophy = fftw_interface<rnumber>::alloc_real((this->cd->local_size/3)*2);
//    this->ift_vorticity();
//    double average_local = 0;
//    RLOOP(
//                this,
//                [&](ptrdiff_t rindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
//        ptrdiff_t tindex = 3*rindex;
//        enstrophy[rindex] = (
//                    this->rvorticity[tindex+0]*this->rvorticity[tindex+0] +
//                this->rvorticity[tindex+1]*this->rvorticity[tindex+1] +
//                this->rvorticity[tindex+2]*this->rvorticity[tindex+2]
//                )/2;
//        average_local += enstrophy[rindex];
//    }
//    );
//    double average;
//    MPI_Allreduce(
//                &average_local,
//                &average,
//                1,
//                MPI_DOUBLE, MPI_SUM, this->cd->comm);
//    DEBUG_MSG("average enstrophy is %g\n", average);
//    /* output goes here */
//    int ntmp[3];
//    ntmp[0] = this->rd->sizes[0];
//    ntmp[1] = this->rd->sizes[1];
//    ntmp[2] = this->rd->sizes[2];
//    field_descriptor<rnumber> *scalar_descriptor = new field_descriptor<rnumber>(3, ntmp, mpi_real_type<rnumber>::real(), this->cd->comm);
//    clip_zero_padding<rnumber>(scalar_descriptor, enstrophy, 1);
//    int return_value = scalar_descriptor->write(fname, enstrophy);
//    delete scalar_descriptor;
//    fftw_interface<rnumber>::free(enstrophy);
//    return return_value;
//}

template <class rnumber,
          field_backend be>
void vorticity_equation<rnumber, be>::compute_pressure(field<rnumber, be, ONE> *pressure)
{
    /* assume velocity is already in real space representation */

    this->v[1]->real_space_representation = true;
    /* diagonal terms 11 22 33 */
    this->v[1]->RLOOP (
                [&](ptrdiff_t rindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex){
        ptrdiff_t tindex = 3*rindex;
        for (int cc=0; cc<3; cc++)
            this->v[1]->get_rdata()[tindex+cc] = this->u->get_rdata()[tindex+cc]*this->u->get_rdata()[tindex+cc];
        }
        );
    //this->clean_up_real_space(this->rv[1], 3);
    this->v[1]->dft();
    this->kk->template dealias<rnumber, THREE>(this->v[1]->get_cdata());
    this->kk->CLOOP_K2(
                [&](ptrdiff_t cindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex,
                    double k2){
        if (k2 <= this->kk->kM2 && k2 > 0)
        {
            ptrdiff_t tindex = 3*cindex;
            for (int i=0; i<2; i++)
            {
                pressure->get_cdata()[cindex][i] = \
                    -(this->kk->kx[xindex]*this->kk->kx[xindex]*this->v[1]->get_cdata()[tindex+0][i] +
                      this->kk->ky[yindex]*this->kk->ky[yindex]*this->v[1]->get_cdata()[tindex+1][i] +
                      this->kk->kz[zindex]*this->kk->kz[zindex]*this->v[1]->get_cdata()[tindex+2][i]);
            }
        }
        else
            std::fill_n((rnumber*)(pressure->get_cdata()+cindex), 2, 0.0);
    }
    );
    /* off-diagonal terms 12 23 31 */
    this->v[1]->real_space_representation = true;
    this->v[1]->RLOOP (
                [&](ptrdiff_t rindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex){
        ptrdiff_t tindex = 3*rindex;
        for (int cc=0; cc<3; cc++)
            this->v[1]->get_rdata()[tindex+cc] = this->u->get_rdata()[tindex+cc]*this->u->get_rdata()[tindex+(cc+1)%3];
    }
    );
    //this->clean_up_real_space(this->rv[1], 3);
    this->v[1]->dft();
    this->kk->template dealias<rnumber, THREE>(this->v[1]->get_cdata());
    this->kk->CLOOP_K2(
                [&](ptrdiff_t cindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex,
                    double k2){
        if (k2 <= this->kk->kM2 && k2 > 0)
        {
            ptrdiff_t tindex = 3*cindex;
            for (int i=0; i<2; i++)
            {
                pressure->get_cdata()[cindex][i] -= \
                    2*(this->kk->kx[xindex]*this->kk->ky[yindex]*this->v[1]->get_cdata()[tindex+0][i] +
                       this->kk->ky[yindex]*this->kk->kz[zindex]*this->v[1]->get_cdata()[tindex+1][i] +
                       this->kk->kz[zindex]*this->kk->kx[xindex]*this->v[1]->get_cdata()[tindex+2][i]);
                pressure->get_cdata()[cindex][i] /= pressure->npoints*k2;
            }
        }
    }
    );
}

//template <class rnumber,
//          field_backend be>
//void vorticity_equation<rnumber, be>::compute_gradient_statistics(
//        rnumber (*__restrict__ vec)[2],
//double *gradu_moments,
//double *trS2QR_moments,
//ptrdiff_t *gradu_hist,
//ptrdiff_t *trS2QR_hist,
//ptrdiff_t *QR2D_hist,
//double trS2QR_max_estimates[],
//double gradu_max_estimates[],
//int nbins,
//int QR2D_nbins)
//{
//    typename fftw_interface<rnumber>::complex *ca;
//    rnumber *ra;
//    ca = fftw_interface<rnumber>::alloc_complex(this->cd->local_size*3);
//    ra = (rnumber*)(ca);
//    this->compute_vector_gradient(ca, vec);
//    for (int cc=0; cc<3; cc++)
//    {
//        std::copy(
//                    (rnumber*)(ca + cc*this->cd->local_size),
//                    (rnumber*)(ca + (cc+1)*this->cd->local_size),
//                    (rnumber*)this->cv[1]);
//        fftw_interface<rnumber>::execute(*(this->vc2r[1]));
//        std::copy(
//                    this->rv[1],
//                this->rv[1] + this->cd->local_size*2,
//                ra + cc*this->cd->local_size*2);
//    }
//    /* velocity gradient is now stored, in real space, in ra */
//    std::fill_n(this->rv[1], 2*this->cd->local_size, 0.0);
//    rnumber *dx_u, *dy_u, *dz_u;
//    dx_u = ra;
//    dy_u = ra + 2*this->cd->local_size;
//    dz_u = ra + 4*this->cd->local_size;
//    double binsize[2];
//    double tmp_max_estimate[3];
//    tmp_max_estimate[0] = trS2QR_max_estimates[0];
//    tmp_max_estimate[1] = trS2QR_max_estimates[1];
//    tmp_max_estimate[2] = trS2QR_max_estimates[2];
//    binsize[0] = 2*tmp_max_estimate[2] / QR2D_nbins;
//    binsize[1] = 2*tmp_max_estimate[1] / QR2D_nbins;
//    ptrdiff_t *local_hist = new ptrdiff_t[QR2D_nbins*QR2D_nbins];
//    std::fill_n(local_hist, QR2D_nbins*QR2D_nbins, 0);
//    RLOOP(
//                this,
//                [&](ptrdiff_t rindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
//        rnumber AxxAxx;
//        rnumber AyyAyy;
//        rnumber AzzAzz;
//        rnumber AxyAyx;
//        rnumber AyzAzy;
//        rnumber AzxAxz;
//        rnumber Sxy;
//        rnumber Syz;
//        rnumber Szx;
//        ptrdiff_t tindex = 3*rindex;
//        AxxAxx = dx_u[tindex+0]*dx_u[tindex+0];
//        AyyAyy = dy_u[tindex+1]*dy_u[tindex+1];
//        AzzAzz = dz_u[tindex+2]*dz_u[tindex+2];
//        AxyAyx = dx_u[tindex+1]*dy_u[tindex+0];
//        AyzAzy = dy_u[tindex+2]*dz_u[tindex+1];
//        AzxAxz = dz_u[tindex+0]*dx_u[tindex+2];
//        this->rv[1][tindex+1] = - (AxxAxx + AyyAyy + AzzAzz)/2 - AxyAyx - AyzAzy - AzxAxz;
//        this->rv[1][tindex+2] = - (dx_u[tindex+0]*(AxxAxx/3 + AxyAyx + AzxAxz) +
//                dy_u[tindex+1]*(AyyAyy/3 + AxyAyx + AyzAzy) +
//                dz_u[tindex+2]*(AzzAzz/3 + AzxAxz + AyzAzy) +
//                dx_u[tindex+1]*dy_u[tindex+2]*dz_u[tindex+0] +
//                dx_u[tindex+2]*dy_u[tindex+0]*dz_u[tindex+1]);
//        int bin0 = int(floor((this->rv[1][tindex+2] + tmp_max_estimate[2]) / binsize[0]));
//        int bin1 = int(floor((this->rv[1][tindex+1] + tmp_max_estimate[1]) / binsize[1]));
//        if ((bin0 >= 0 && bin0 < QR2D_nbins) &&
//                (bin1 >= 0 && bin1 < QR2D_nbins))
//            local_hist[bin1*QR2D_nbins + bin0]++;
//        Sxy = dx_u[tindex+1]+dy_u[tindex+0];
//        Syz = dy_u[tindex+2]+dz_u[tindex+1];
//        Szx = dz_u[tindex+0]+dx_u[tindex+2];
//        this->rv[1][tindex] = (AxxAxx + AyyAyy + AzzAzz +
//                               (Sxy*Sxy + Syz*Syz + Szx*Szx)/2);
//    }
//    );
//    MPI_Allreduce(
//                local_hist,
//                QR2D_hist,
//                QR2D_nbins * QR2D_nbins,
//                MPI_INT64_T, MPI_SUM, this->cd->comm);
//    delete[] local_hist;
//    this->compute_rspace_stats3(
//                this->rv[1],
//            trS2QR_moments,
//            trS2QR_hist,
//            tmp_max_estimate,
//            nbins);
//    double *tmp_moments = new double[10*3];
//    ptrdiff_t *tmp_hist = new ptrdiff_t[nbins*3];
//    for (int cc=0; cc<3; cc++)
//    {
//        tmp_max_estimate[0] = gradu_max_estimates[cc*3 + 0];
//        tmp_max_estimate[1] = gradu_max_estimates[cc*3 + 1];
//        tmp_max_estimate[2] = gradu_max_estimates[cc*3 + 2];
//        this->compute_rspace_stats3(
//                    dx_u + cc*2*this->cd->local_size,
//                    tmp_moments,
//                    tmp_hist,
//                    tmp_max_estimate,
//                    nbins);
//        for (int n = 0; n < 10; n++)
//            for (int i = 0; i < 3 ; i++)
//            {
//                gradu_moments[(n*3 + cc)*3 + i] = tmp_moments[n*3 + i];
//            }
//        for (int n = 0; n < nbins; n++)
//            for (int i = 0; i < 3; i++)
//            {
//                gradu_hist[(n*3 + cc)*3 + i] = tmp_hist[n*3 + i];
//            }
//    }
//    delete[] tmp_moments;
//    delete[] tmp_hist;
//    fftw_interface<rnumber>::free(ca);
//}
//
//template <class rnumber,
//          field_backend be>
//void vorticity_equation<rnumber, be>::compute_Lagrangian_acceleration(rnumber (*acceleration)[2])
//{
//    ptrdiff_t tindex;
//    typename fftw_interface<rnumber>::complex *pressure;
//    pressure = fftw_interface<rnumber>::alloc_complex(this->cd->local_size/3);
//    this->compute_velocity(this->cvorticity);
//    this->ift_velocity();
//    this->compute_pressure(pressure);
//    this->compute_velocity(this->cvorticity);
//    std::fill_n((rnumber*)this->cv[1], 2*this->cd->local_size, 0.0);
//    CLOOP_K2(
//                this,
//                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
//        if (k2 <= this->kM2)
//        {
//            tindex = 3*cindex;
//            for (int cc=0; cc<3; cc++)
//                for (int i=0; i<2; i++)
//                    this->cv[1][tindex+cc][i] = - this->nu*k2*this->cu[tindex+cc][i];
//            if (strcmp(this->forcing_type, "linear") == 0)
//            {
//                double knorm = sqrt(k2);
//                if ((this->fk0 <= knorm) &&
//                        (this->fk1 >= knorm))
//                    for (int c=0; c<3; c++)
//                        for (int i=0; i<2; i++)
//                            this->cv[1][tindex+c][i] += this->famplitude*this->cu[tindex+c][i];
//            }
//            this->cv[1][tindex+0][0] += this->kx[xindex]*pressure[cindex][1];
//            this->cv[1][tindex+1][0] += this->ky[yindex]*pressure[cindex][1];
//            this->cv[1][tindex+2][0] += this->kz[zindex]*pressure[cindex][1];
//            this->cv[1][tindex+0][1] -= this->kx[xindex]*pressure[cindex][0];
//            this->cv[1][tindex+1][1] -= this->ky[yindex]*pressure[cindex][0];
//            this->cv[1][tindex+2][1] -= this->kz[zindex]*pressure[cindex][0];
//        }
//    }
//    );
//    std::copy(
//                (rnumber*)this->cv[1],
//            (rnumber*)(this->cv[1] + this->cd->local_size),
//            (rnumber*)acceleration);
//    fftw_interface<rnumber>::free(pressure);
//}

//template <class rnumber,
//          field_backend be>
//void vorticity_equation<rnumber, be>::compute_Eulerian_acceleration(rnumber (*__restrict__ acceleration)[2])
//{
//    std::fill_n((rnumber*)(acceleration), 2*this->cd->local_size, 0.0);
//    this->compute_velocity(this->cvorticity);
//    /* put in linear terms */
//    this->kk->CLOOP_K2(
//                [&](ptrdiff_t cindex,
//                    ptrdiff_t xindex,
//                    ptrdiff_t yindex,
//                    ptrdiff_t zindex,
//                    double k2){
//        if (k2 <= this->kk->kM2)
//        {
//            ptrdiff_t tindex = 3*cindex;
//            for (int cc=0; cc<3; cc++)
//                for (int i=0; i<2; i++)
//                    acceleration[tindex+cc][i] = - this->nu*k2*this->cu[tindex+cc][i];
//            if (strcmp(this->forcing_type, "linear") == 0)
//            {
//                double knorm = sqrt(k2);
//                if ((this->fk0 <= knorm) &&
//                        (this->fk1 >= knorm))
//                {
//                    for (int c=0; c<3; c++)
//                        for (int i=0; i<2; i++)
//                            acceleration[tindex+c][i] += this->famplitude*this->cu[tindex+c][i];
//                }
//            }
//        }
//    }
//    );
//    this->ift_velocity();
//    /* compute uu */
//    /* 11 22 33 */
//    RLOOP (
//                this,
//                [&](ptrdiff_t rindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
//        tindex = 3*rindex;
//        for (int cc=0; cc<3; cc++)
//            this->rv[1][tindex+cc] = this->ru[tindex+cc]*this->ru[tindex+cc] / this->normalization_factor;
//    }
//    );
//    this->clean_up_real_space(this->rv[1], 3);
//    fftw_interface<rnumber>::execute(*(this->vr2c[1]));
//    this->dealias(this->cv[1], 3);
//    CLOOP_K2(
//                this,
//                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
//        if (k2 <= this->kM2)
//        {
//            tindex = 3*cindex;
//            acceleration[tindex+0][0] +=
//                    this->kx[xindex]*this->cv[1][tindex+0][1];
//            acceleration[tindex+0][1] +=
//                    -this->kx[xindex]*this->cv[1][tindex+0][0];
//            acceleration[tindex+1][0] +=
//                    this->ky[yindex]*this->cv[1][tindex+1][1];
//            acceleration[tindex+1][1] +=
//                    -this->ky[yindex]*this->cv[1][tindex+1][0];
//            acceleration[tindex+2][0] +=
//                    this->kz[zindex]*this->cv[1][tindex+2][1];
//            acceleration[tindex+2][1] +=
//                    -this->kz[zindex]*this->cv[1][tindex+2][0];
//        }
//    }
//    );
//    /* 12 23 31 */
//    RLOOP (
//                this,
//                [&](ptrdiff_t rindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex){
//        tindex = 3*rindex;
//        for (int cc=0; cc<3; cc++)
//            this->rv[1][tindex+cc] = this->ru[tindex+cc]*this->ru[tindex+(cc+1)%3] / this->normalization_factor;
//    }
//    );
//    this->clean_up_real_space(this->rv[1], 3);
//    fftw_interface<rnumber>::execute(*(this->vr2c[1]));
//    this->dealias(this->cv[1], 3);
//    CLOOP_K2(
//                this,
//                [&](ptrdiff_t cindex, ptrdiff_t xindex, ptrdiff_t yindex, ptrdiff_t zindex, double k2){
//        if (k2 <= this->kM2)
//        {
//            tindex = 3*cindex;
//            acceleration[tindex+0][0] +=
//                    (this->ky[yindex]*this->cv[1][tindex+0][1] +
//                    this->kz[zindex]*this->cv[1][tindex+2][1]);
//            acceleration[tindex+0][1] +=
//                    - (this->ky[yindex]*this->cv[1][tindex+0][0] +
//                    this->kz[zindex]*this->cv[1][tindex+2][0]);
//            acceleration[tindex+1][0] +=
//                    (this->kz[zindex]*this->cv[1][tindex+1][1] +
//                    this->kx[xindex]*this->cv[1][tindex+0][1]);
//            acceleration[tindex+1][1] +=
//                    - (this->kz[zindex]*this->cv[1][tindex+1][0] +
//                    this->kx[xindex]*this->cv[1][tindex+0][0]);
//            acceleration[tindex+2][0] +=
//                    (this->kx[xindex]*this->cv[1][tindex+2][1] +
//                    this->ky[yindex]*this->cv[1][tindex+1][1]);
//            acceleration[tindex+2][1] +=
//                    - (this->kx[xindex]*this->cv[1][tindex+2][0] +
//                    this->ky[yindex]*this->cv[1][tindex+1][0]);
//        }
//    }
//    );
//    if (this->cd->myrank == this->cd->rank[0])
//        std::fill_n((rnumber*)(acceleration), 6, 0.0);
//    this->force_divfree(acceleration);
//}
//
//template <class rnumber,
//          field_backend be>
//void vorticity_equation<rnumber, be>::compute_Lagrangian_acceleration(rnumber *__restrict__ acceleration)
//{
//    this->compute_Lagrangian_acceleration((typename fftw_interface<rnumber>::complex*)acceleration);
//    fftw_interface<rnumber>::execute(*(this->vc2r[1]));
//    std::copy(
//                this->rv[1],
//            this->rv[1] + 2*this->cd->local_size,
//            acceleration);
//}
//
//template <class rnumber,
//          field_backend be>
//int vorticity_equation<rnumber, be>::write_rpressure()
//{
//    char fname[512];
//    typename fftw_interface<rnumber>::complex *pressure;
//    pressure = fftw_interface<rnumber>::alloc_complex(this->cd->local_size/3);
//    this->compute_velocity(this->cvorticity);
//    this->ift_velocity();
//    this->compute_pressure(pressure);
//    this->fill_up_filename("rpressure", fname);
//    rnumber *rpressure = fftw_interface<rnumber>::alloc_real((this->cd->local_size/3)*2);
//    typename fftw_interface<rnumber>::plan c2r;
//    c2r = fftw_interface<rnumber>::mpi_plan_dft_c2r_3d(
//                this->rd->sizes[0], this->rd->sizes[1], this->rd->sizes[2],
//            pressure, rpressure, this->cd->comm,
//            this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_IN);
//    fftw_interface<rnumber>::execute(c2r);
//    /* output goes here */
//    int ntmp[3];
//    ntmp[0] = this->rd->sizes[0];
//    ntmp[1] = this->rd->sizes[1];
//    ntmp[2] = this->rd->sizes[2];
//    field_descriptor<rnumber> *scalar_descriptor = new field_descriptor<rnumber>(3, ntmp, mpi_real_type<rnumber>::real(), this->cd->comm);
//    clip_zero_padding<rnumber>(scalar_descriptor, rpressure, 1);
//    int return_value = scalar_descriptor->write(fname, rpressure);
//    delete scalar_descriptor;
//    fftw_interface<rnumber>::destroy_plan(c2r);
//    fftw_interface<rnumber>::free(pressure);
//    fftw_interface<rnumber>::free(rpressure);
//    return return_value;
//}

/*****************************************************************************/




/*****************************************************************************/
/* finally, force generation of code for single precision                    */
template class vorticity_equation<float, FFTW>;
template class vorticity_equation<double, FFTW>;
/*****************************************************************************/

