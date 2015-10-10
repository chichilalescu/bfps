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



#include "spline_n1.hpp"
#include <math.h>

void beta_n1_m0(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = 0;
        poly_val[1] = -x + 1;
        poly_val[2] = x;
        poly_val[3] = 0;
        break;
    case 1:
        poly_val[0] = 0;
        poly_val[1] = -1;
        poly_val[2] = 1;
        poly_val[3] = 0;
        break;
    case 2:
        poly_val[0] = 0;
        poly_val[1] = 0;
        poly_val[2] = 0;
        poly_val[3] = 0;
        break;
    }
}

void beta_n1_m1(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = x*(x*(-1.0L/2.0L*x + 1) - 1.0L/2.0L);
        poly_val[1] = pow(x, 2)*((3.0L/2.0L)*x - 5.0L/2.0L) + 1;
        poly_val[2] = x*(x*(-3.0L/2.0L*x + 2) + 1.0L/2.0L);
        poly_val[3] = pow(x, 2)*((1.0L/2.0L)*x - 1.0L/2.0L);
        break;
    case 1:
        poly_val[0] = x*(-3.0L/2.0L*x + 2) - 1.0L/2.0L;
        poly_val[1] = x*((9.0L/2.0L)*x - 5);
        poly_val[2] = x*(-9.0L/2.0L*x + 4) + 1.0L/2.0L;
        poly_val[3] = x*((3.0L/2.0L)*x - 1);
        break;
    case 2:
        poly_val[0] = -3*x + 2;
        poly_val[1] = 9*x - 5;
        poly_val[2] = -9*x + 4;
        poly_val[3] = 3*x - 1;
        break;
    }
}

void beta_n1_m2(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = x*(x*(x*(x*(x - 5.0L/2.0L) + 3.0L/2.0L) + 1.0L/2.0L) - 1.0L/2.0L);
        poly_val[1] = pow(x, 2)*(x*(x*(-3*x + 15.0L/2.0L) - 9.0L/2.0L) - 1) + 1;
        poly_val[2] = x*(x*(x*(x*(3*x - 15.0L/2.0L) + 9.0L/2.0L) + 1.0L/2.0L) + 1.0L/2.0L);
        poly_val[3] = pow(x, 3)*(x*(-x + 5.0L/2.0L) - 3.0L/2.0L);
        break;
    case 1:
        poly_val[0] = x*(x*(x*(5*x - 10) + 9.0L/2.0L) + 1) - 1.0L/2.0L;
        poly_val[1] = x*(x*(x*(-15*x + 30) - 27.0L/2.0L) - 2);
        poly_val[2] = x*(x*(x*(15*x - 30) + 27.0L/2.0L) + 1) + 1.0L/2.0L;
        poly_val[3] = pow(x, 2)*(x*(-5*x + 10) - 9.0L/2.0L);
        break;
    case 2:
        poly_val[0] = x*(x*(20*x - 30) + 9) + 1;
        poly_val[1] = x*(x*(-60*x + 90) - 27) - 2;
        poly_val[2] = x*(x*(60*x - 90) + 27) + 1;
        poly_val[3] = x*(x*(-20*x + 30) - 9);
        break;
    }
}

