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



#include "spline_n2.hpp"
#include <math.h>

void beta_n2_m0(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = 0;
        poly_val[1] = 0;
        poly_val[2] = -x + 1;
        poly_val[3] = x;
        poly_val[4] = 0;
        poly_val[5] = 0;
        break;
    case 1:
        poly_val[0] = 0;
        poly_val[1] = 0;
        poly_val[2] = -1;
        poly_val[3] = 1;
        poly_val[4] = 0;
        poly_val[5] = 0;
        break;
    case 2:
        poly_val[0] = 0;
        poly_val[1] = 0;
        poly_val[2] = 0;
        poly_val[3] = 0;
        poly_val[4] = 0;
        poly_val[5] = 0;
        break;
    }
}

void beta_n2_m1(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = x*(x*((1.0L/12.0L)*x - 1.0L/6.0L) + 1.0L/12.0L);
        poly_val[1] = x*(x*(-7.0L/12.0L*x + 5.0L/4.0L) - 2.0L/3.0L);
        poly_val[2] = pow(x, 2)*((4.0L/3.0L)*x - 7.0L/3.0L) + 1;
        poly_val[3] = x*(x*(-4.0L/3.0L*x + 5.0L/3.0L) + 2.0L/3.0L);
        poly_val[4] = x*(x*((7.0L/12.0L)*x - 1.0L/2.0L) - 1.0L/12.0L);
        poly_val[5] = pow(x, 2)*(-1.0L/12.0L*x + 1.0L/12.0L);
        break;
    case 1:
        poly_val[0] = x*((1.0L/4.0L)*x - 1.0L/3.0L) + 1.0L/12.0L;
        poly_val[1] = x*(-7.0L/4.0L*x + 5.0L/2.0L) - 2.0L/3.0L;
        poly_val[2] = x*(4*x - 14.0L/3.0L);
        poly_val[3] = x*(-4*x + 10.0L/3.0L) + 2.0L/3.0L;
        poly_val[4] = x*((7.0L/4.0L)*x - 1) - 1.0L/12.0L;
        poly_val[5] = x*(-1.0L/4.0L*x + 1.0L/6.0L);
        break;
    case 2:
        poly_val[0] = (1.0L/2.0L)*x - 1.0L/3.0L;
        poly_val[1] = -7.0L/2.0L*x + 5.0L/2.0L;
        poly_val[2] = 8*x - 14.0L/3.0L;
        poly_val[3] = -8*x + 10.0L/3.0L;
        poly_val[4] = (7.0L/2.0L)*x - 1;
        poly_val[5] = -1.0L/2.0L*x + 1.0L/6.0L;
        break;
    }
}

void beta_n2_m2(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = x*(x*(x*(x*(-5.0L/24.0L*x + 13.0L/24.0L) - 3.0L/8.0L) - 1.0L/24.0L) + 1.0L/12.0L);
        poly_val[1] = x*(x*(x*(x*((25.0L/24.0L)*x - 8.0L/3.0L) + 13.0L/8.0L) + 2.0L/3.0L) - 2.0L/3.0L);
        poly_val[2] = pow(x, 2)*(x*(x*(-25.0L/12.0L*x + 21.0L/4.0L) - 35.0L/12.0L) - 5.0L/4.0L) + 1;
        poly_val[3] = x*(x*(x*(x*((25.0L/12.0L)*x - 31.0L/6.0L) + 11.0L/4.0L) + 2.0L/3.0L) + 2.0L/3.0L);
        poly_val[4] = x*(x*(x*(x*(-25.0L/24.0L*x + 61.0L/24.0L) - 11.0L/8.0L) - 1.0L/24.0L) - 1.0L/12.0L);
        poly_val[5] = pow(x, 3)*(x*((5.0L/24.0L)*x - 1.0L/2.0L) + 7.0L/24.0L);
        break;
    case 1:
        poly_val[0] = x*(x*(x*(-25.0L/24.0L*x + 13.0L/6.0L) - 9.0L/8.0L) - 1.0L/12.0L) + 1.0L/12.0L;
        poly_val[1] = x*(x*(x*((125.0L/24.0L)*x - 32.0L/3.0L) + 39.0L/8.0L) + 4.0L/3.0L) - 2.0L/3.0L;
        poly_val[2] = x*(x*(x*(-125.0L/12.0L*x + 21) - 35.0L/4.0L) - 5.0L/2.0L);
        poly_val[3] = x*(x*(x*((125.0L/12.0L)*x - 62.0L/3.0L) + 33.0L/4.0L) + 4.0L/3.0L) + 2.0L/3.0L;
        poly_val[4] = x*(x*(x*(-125.0L/24.0L*x + 61.0L/6.0L) - 33.0L/8.0L) - 1.0L/12.0L) - 1.0L/12.0L;
        poly_val[5] = pow(x, 2)*(x*((25.0L/24.0L)*x - 2) + 7.0L/8.0L);
        break;
    case 2:
        poly_val[0] = x*(x*(-25.0L/6.0L*x + 13.0L/2.0L) - 9.0L/4.0L) - 1.0L/12.0L;
        poly_val[1] = x*(x*((125.0L/6.0L)*x - 32) + 39.0L/4.0L) + 4.0L/3.0L;
        poly_val[2] = x*(x*(-125.0L/3.0L*x + 63) - 35.0L/2.0L) - 5.0L/2.0L;
        poly_val[3] = x*(x*((125.0L/3.0L)*x - 62) + 33.0L/2.0L) + 4.0L/3.0L;
        poly_val[4] = x*(x*(-125.0L/6.0L*x + 61.0L/2.0L) - 33.0L/4.0L) - 1.0L/12.0L;
        poly_val[5] = x*(x*((25.0L/6.0L)*x - 6) + 7.0L/4.0L);
        break;
    }
}

void beta_n2_m3(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = x*(x*(x*(x*(x*(x*((7.0L/12.0L)*x - 49.0L/24.0L) + 29.0L/12.0L) - 11.0L/12.0L) - 1.0L/12.0L) - 1.0L/24.0L) + 1.0L/12.0L);
        poly_val[1] = x*(x*(x*(x*(x*(x*(-35.0L/12.0L*x + 245.0L/24.0L) - 145.0L/12.0L) + 37.0L/8.0L) + 1.0L/6.0L) + 2.0L/3.0L) - 2.0L/3.0L);
        poly_val[2] = pow(x, 2)*(pow(x, 2)*(x*(x*((35.0L/6.0L)*x - 245.0L/12.0L) + 145.0L/6.0L) - 28.0L/3.0L) - 5.0L/4.0L) + 1;
        poly_val[3] = x*(x*(x*(x*(x*(x*(-35.0L/6.0L*x + 245.0L/12.0L) - 145.0L/6.0L) + 113.0L/12.0L) - 1.0L/6.0L) + 2.0L/3.0L) + 2.0L/3.0L);
        poly_val[4] = x*(x*(x*(x*(x*(x*((35.0L/12.0L)*x - 245.0L/24.0L) + 145.0L/12.0L) - 19.0L/4.0L) + 1.0L/12.0L) - 1.0L/24.0L) - 1.0L/12.0L);
        poly_val[5] = pow(x, 4)*(x*(x*(-7.0L/12.0L*x + 49.0L/24.0L) - 29.0L/12.0L) + 23.0L/24.0L);
        break;
    case 1:
        poly_val[0] = x*(x*(x*(x*(x*((49.0L/12.0L)*x - 49.0L/4.0L) + 145.0L/12.0L) - 11.0L/3.0L) - 1.0L/4.0L) - 1.0L/12.0L) + 1.0L/12.0L;
        poly_val[1] = x*(x*(x*(x*(x*(-245.0L/12.0L*x + 245.0L/4.0L) - 725.0L/12.0L) + 37.0L/2.0L) + 1.0L/2.0L) + 4.0L/3.0L) - 2.0L/3.0L;
        poly_val[2] = x*(pow(x, 2)*(x*(x*((245.0L/6.0L)*x - 245.0L/2.0L) + 725.0L/6.0L) - 112.0L/3.0L) - 5.0L/2.0L);
        poly_val[3] = x*(x*(x*(x*(x*(-245.0L/6.0L*x + 245.0L/2.0L) - 725.0L/6.0L) + 113.0L/3.0L) - 1.0L/2.0L) + 4.0L/3.0L) + 2.0L/3.0L;
        poly_val[4] = x*(x*(x*(x*(x*((245.0L/12.0L)*x - 245.0L/4.0L) + 725.0L/12.0L) - 19) + 1.0L/4.0L) - 1.0L/12.0L) - 1.0L/12.0L;
        poly_val[5] = pow(x, 3)*(x*(x*(-49.0L/12.0L*x + 49.0L/4.0L) - 145.0L/12.0L) + 23.0L/6.0L);
        break;
    case 2:
        poly_val[0] = x*(x*(x*(x*((49.0L/2.0L)*x - 245.0L/4.0L) + 145.0L/3.0L) - 11) - 1.0L/2.0L) - 1.0L/12.0L;
        poly_val[1] = x*(x*(x*(x*(-245.0L/2.0L*x + 1225.0L/4.0L) - 725.0L/3.0L) + 111.0L/2.0L) + 1) + 4.0L/3.0L;
        poly_val[2] = pow(x, 2)*(x*(x*(245*x - 1225.0L/2.0L) + 1450.0L/3.0L) - 112) - 5.0L/2.0L;
        poly_val[3] = x*(x*(x*(x*(-245*x + 1225.0L/2.0L) - 1450.0L/3.0L) + 113) - 1) + 4.0L/3.0L;
        poly_val[4] = x*(x*(x*(x*((245.0L/2.0L)*x - 1225.0L/4.0L) + 725.0L/3.0L) - 57) + 1.0L/2.0L) - 1.0L/12.0L;
        poly_val[5] = pow(x, 2)*(x*(x*(-49.0L/2.0L*x + 245.0L/4.0L) - 145.0L/3.0L) + 23.0L/2.0L);
        break;
    }
}

void beta_n2_m4(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = x*(x*(x*(x*(x*(x*(x*(x*(-23.0L/12.0L*x + 69.0L/8.0L) - 59.0L/4.0L) + 91.0L/8.0L) - 10.0L/3.0L) + 1.0L/24.0L) - 1.0L/12.0L) - 1.0L/24.0L) + 1.0L/12.0L);
        poly_val[1] = x*(x*(x*(x*(x*(x*(x*(x*((115.0L/12.0L)*x - 345.0L/8.0L) + 295.0L/4.0L) - 455.0L/8.0L) + 50.0L/3.0L) - 1.0L/6.0L) + 1.0L/6.0L) + 2.0L/3.0L) - 2.0L/3.0L);
        poly_val[2] = pow(x, 2)*(pow(x, 2)*(x*(x*(x*(x*(-115.0L/6.0L*x + 345.0L/4.0L) - 295.0L/2.0L) + 455.0L/4.0L) - 100.0L/3.0L) + 1.0L/4.0L) - 5.0L/4.0L) + 1;
        poly_val[3] = x*(x*(x*(x*(x*(x*(x*(x*((115.0L/6.0L)*x - 345.0L/4.0L) + 295.0L/2.0L) - 455.0L/4.0L) + 100.0L/3.0L) - 1.0L/6.0L) - 1.0L/6.0L) + 2.0L/3.0L) + 2.0L/3.0L);
        poly_val[4] = x*(x*(x*(x*(x*(x*(x*(x*(-115.0L/12.0L*x + 345.0L/8.0L) - 295.0L/4.0L) + 455.0L/8.0L) - 50.0L/3.0L) + 1.0L/24.0L) + 1.0L/12.0L) - 1.0L/24.0L) - 1.0L/12.0L);
        poly_val[5] = pow(x, 5)*(x*(x*(x*((23.0L/12.0L)*x - 69.0L/8.0L) + 59.0L/4.0L) - 91.0L/8.0L) + 10.0L/3.0L);
        break;
    case 1:
        poly_val[0] = x*(x*(x*(x*(x*(x*(x*(-69.0L/4.0L*x + 69) - 413.0L/4.0L) + 273.0L/4.0L) - 50.0L/3.0L) + 1.0L/6.0L) - 1.0L/4.0L) - 1.0L/12.0L) + 1.0L/12.0L;
        poly_val[1] = x*(x*(x*(x*(x*(x*(x*((345.0L/4.0L)*x - 345) + 2065.0L/4.0L) - 1365.0L/4.0L) + 250.0L/3.0L) - 2.0L/3.0L) + 1.0L/2.0L) + 4.0L/3.0L) - 2.0L/3.0L;
        poly_val[2] = x*(pow(x, 2)*(x*(x*(x*(x*(-345.0L/2.0L*x + 690) - 2065.0L/2.0L) + 1365.0L/2.0L) - 500.0L/3.0L) + 1) - 5.0L/2.0L);
        poly_val[3] = x*(x*(x*(x*(x*(x*(x*((345.0L/2.0L)*x - 690) + 2065.0L/2.0L) - 1365.0L/2.0L) + 500.0L/3.0L) - 2.0L/3.0L) - 1.0L/2.0L) + 4.0L/3.0L) + 2.0L/3.0L;
        poly_val[4] = x*(x*(x*(x*(x*(x*(x*(-345.0L/4.0L*x + 345) - 2065.0L/4.0L) + 1365.0L/4.0L) - 250.0L/3.0L) + 1.0L/6.0L) + 1.0L/4.0L) - 1.0L/12.0L) - 1.0L/12.0L;
        poly_val[5] = pow(x, 4)*(x*(x*(x*((69.0L/4.0L)*x - 69) + 413.0L/4.0L) - 273.0L/4.0L) + 50.0L/3.0L);
        break;
    case 2:
        poly_val[0] = x*(x*(x*(x*(x*(x*(-138*x + 483) - 1239.0L/2.0L) + 1365.0L/4.0L) - 200.0L/3.0L) + 1.0L/2.0L) - 1.0L/2.0L) - 1.0L/12.0L;
        poly_val[1] = x*(x*(x*(x*(x*(x*(690*x - 2415) + 6195.0L/2.0L) - 6825.0L/4.0L) + 1000.0L/3.0L) - 2) + 1) + 4.0L/3.0L;
        poly_val[2] = pow(x, 2)*(x*(x*(x*(x*(-1380*x + 4830) - 6195) + 6825.0L/2.0L) - 2000.0L/3.0L) + 3) - 5.0L/2.0L;
        poly_val[3] = x*(x*(x*(x*(x*(x*(1380*x - 4830) + 6195) - 6825.0L/2.0L) + 2000.0L/3.0L) - 2) - 1) + 4.0L/3.0L;
        poly_val[4] = x*(x*(x*(x*(x*(x*(-690*x + 2415) - 6195.0L/2.0L) + 6825.0L/4.0L) - 1000.0L/3.0L) + 1.0L/2.0L) + 1.0L/2.0L) - 1.0L/12.0L;
        poly_val[5] = pow(x, 3)*(x*(x*(x*(138*x - 483) + 1239.0L/2.0L) - 1365.0L/4.0L) + 200.0L/3.0L);
        break;
    }
}

