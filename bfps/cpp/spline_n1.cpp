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

