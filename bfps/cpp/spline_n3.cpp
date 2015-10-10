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



#include "spline_n3.hpp"
#include <math.h>

void beta_n3_m0(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = 0;
        poly_val[1] = 0;
        poly_val[2] = 0;
        poly_val[3] = -x + 1;
        poly_val[4] = x;
        poly_val[5] = 0;
        poly_val[6] = 0;
        poly_val[7] = 0;
        break;
    case 1:
        poly_val[0] = 0;
        poly_val[1] = 0;
        poly_val[2] = 0;
        poly_val[3] = -1;
        poly_val[4] = 1;
        poly_val[5] = 0;
        poly_val[6] = 0;
        poly_val[7] = 0;
        break;
    case 2:
        poly_val[0] = 0;
        poly_val[1] = 0;
        poly_val[2] = 0;
        poly_val[3] = 0;
        poly_val[4] = 0;
        poly_val[5] = 0;
        poly_val[6] = 0;
        poly_val[7] = 0;
        break;
    }
}

void beta_n3_m1(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = x*(x*(-1.0L/60.0L*x + 1.0L/30.0L) - 1.0L/60.0L);
        poly_val[1] = x*(x*((2.0L/15.0L)*x - 17.0L/60.0L) + 3.0L/20.0L);
        poly_val[2] = x*(x*(-3.0L/5.0L*x + 27.0L/20.0L) - 3.0L/4.0L);
        poly_val[3] = pow(x, 2)*((5.0L/4.0L)*x - 9.0L/4.0L) + 1;
        poly_val[4] = x*(x*(-5.0L/4.0L*x + 3.0L/2.0L) + 3.0L/4.0L);
        poly_val[5] = x*(x*((3.0L/5.0L)*x - 9.0L/20.0L) - 3.0L/20.0L);
        poly_val[6] = x*(x*(-2.0L/15.0L*x + 7.0L/60.0L) + 1.0L/60.0L);
        poly_val[7] = pow(x, 2)*((1.0L/60.0L)*x - 1.0L/60.0L);
        break;
    case 1:
        poly_val[0] = x*(-1.0L/20.0L*x + 1.0L/15.0L) - 1.0L/60.0L;
        poly_val[1] = x*((2.0L/5.0L)*x - 17.0L/30.0L) + 3.0L/20.0L;
        poly_val[2] = x*(-9.0L/5.0L*x + 27.0L/10.0L) - 3.0L/4.0L;
        poly_val[3] = x*((15.0L/4.0L)*x - 9.0L/2.0L);
        poly_val[4] = x*(-15.0L/4.0L*x + 3) + 3.0L/4.0L;
        poly_val[5] = x*((9.0L/5.0L)*x - 9.0L/10.0L) - 3.0L/20.0L;
        poly_val[6] = x*(-2.0L/5.0L*x + 7.0L/30.0L) + 1.0L/60.0L;
        poly_val[7] = x*((1.0L/20.0L)*x - 1.0L/30.0L);
        break;
    case 2:
        poly_val[0] = -1.0L/10.0L*x + 1.0L/15.0L;
        poly_val[1] = (4.0L/5.0L)*x - 17.0L/30.0L;
        poly_val[2] = -18.0L/5.0L*x + 27.0L/10.0L;
        poly_val[3] = (15.0L/2.0L)*x - 9.0L/2.0L;
        poly_val[4] = -15.0L/2.0L*x + 3;
        poly_val[5] = (18.0L/5.0L)*x - 9.0L/10.0L;
        poly_val[6] = -4.0L/5.0L*x + 7.0L/30.0L;
        poly_val[7] = (1.0L/10.0L)*x - 1.0L/30.0L;
        break;
    }
}

void beta_n3_m2(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = x*(x*(x*(x*((2.0L/45.0L)*x - 7.0L/60.0L) + 1.0L/12.0L) + 1.0L/180.0L) - 1.0L/60.0L);
        poly_val[1] = x*(x*(x*(x*(-23.0L/72.0L*x + 61.0L/72.0L) - 217.0L/360.0L) - 3.0L/40.0L) + 3.0L/20.0L);
        poly_val[2] = x*(x*(x*(x*((39.0L/40.0L)*x - 51.0L/20.0L) + 63.0L/40.0L) + 3.0L/4.0L) - 3.0L/4.0L);
        poly_val[3] = pow(x, 2)*(x*(x*(-59.0L/36.0L*x + 25.0L/6.0L) - 13.0L/6.0L) - 49.0L/36.0L) + 1;
        poly_val[4] = x*(x*(x*(x*((59.0L/36.0L)*x - 145.0L/36.0L) + 17.0L/9.0L) + 3.0L/4.0L) + 3.0L/4.0L);
        poly_val[5] = x*(x*(x*(x*(-39.0L/40.0L*x + 93.0L/40.0L) - 9.0L/8.0L) - 3.0L/40.0L) - 3.0L/20.0L);
        poly_val[6] = x*(x*(x*(x*((23.0L/72.0L)*x - 3.0L/4.0L) + 49.0L/120.0L) + 1.0L/180.0L) + 1.0L/60.0L);
        poly_val[7] = pow(x, 3)*(x*(-2.0L/45.0L*x + 19.0L/180.0L) - 11.0L/180.0L);
        break;
    case 1:
        poly_val[0] = x*(x*(x*((2.0L/9.0L)*x - 7.0L/15.0L) + 1.0L/4.0L) + 1.0L/90.0L) - 1.0L/60.0L;
        poly_val[1] = x*(x*(x*(-115.0L/72.0L*x + 61.0L/18.0L) - 217.0L/120.0L) - 3.0L/20.0L) + 3.0L/20.0L;
        poly_val[2] = x*(x*(x*((39.0L/8.0L)*x - 51.0L/5.0L) + 189.0L/40.0L) + 3.0L/2.0L) - 3.0L/4.0L;
        poly_val[3] = x*(x*(x*(-295.0L/36.0L*x + 50.0L/3.0L) - 13.0L/2.0L) - 49.0L/18.0L);
        poly_val[4] = x*(x*(x*((295.0L/36.0L)*x - 145.0L/9.0L) + 17.0L/3.0L) + 3.0L/2.0L) + 3.0L/4.0L;
        poly_val[5] = x*(x*(x*(-39.0L/8.0L*x + 93.0L/10.0L) - 27.0L/8.0L) - 3.0L/20.0L) - 3.0L/20.0L;
        poly_val[6] = x*(x*(x*((115.0L/72.0L)*x - 3) + 49.0L/40.0L) + 1.0L/90.0L) + 1.0L/60.0L;
        poly_val[7] = pow(x, 2)*(x*(-2.0L/9.0L*x + 19.0L/45.0L) - 11.0L/60.0L);
        break;
    case 2:
        poly_val[0] = x*(x*((8.0L/9.0L)*x - 7.0L/5.0L) + 1.0L/2.0L) + 1.0L/90.0L;
        poly_val[1] = x*(x*(-115.0L/18.0L*x + 61.0L/6.0L) - 217.0L/60.0L) - 3.0L/20.0L;
        poly_val[2] = x*(x*((39.0L/2.0L)*x - 153.0L/5.0L) + 189.0L/20.0L) + 3.0L/2.0L;
        poly_val[3] = x*(x*(-295.0L/9.0L*x + 50) - 13) - 49.0L/18.0L;
        poly_val[4] = x*(x*((295.0L/9.0L)*x - 145.0L/3.0L) + 34.0L/3.0L) + 3.0L/2.0L;
        poly_val[5] = x*(x*(-39.0L/2.0L*x + 279.0L/10.0L) - 27.0L/4.0L) - 3.0L/20.0L;
        poly_val[6] = x*(x*((115.0L/18.0L)*x - 9) + 49.0L/20.0L) + 1.0L/90.0L;
        poly_val[7] = x*(x*(-8.0L/9.0L*x + 19.0L/15.0L) - 11.0L/30.0L);
        break;
    }
}

void beta_n3_m3(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = x*(x*(x*(x*(x*(x*(-89.0L/720.0L*x + 13.0L/30.0L) - 37.0L/72.0L) + 7.0L/36.0L) + 1.0L/48.0L) + 1.0L/180.0L) - 1.0L/60.0L);
        poly_val[1] = x*(x*(x*(x*(x*(x*((623.0L/720.0L)*x - 2183.0L/720.0L) + 2581.0L/720.0L) - 191.0L/144.0L) - 1.0L/6.0L) - 3.0L/40.0L) + 3.0L/20.0L);
        poly_val[2] = x*(x*(x*(x*(x*(x*(-623.0L/240.0L*x + 1091.0L/120.0L) - 429.0L/40.0L) + 95.0L/24.0L) + 13.0L/48.0L) + 3.0L/4.0L) - 3.0L/4.0L);
        poly_val[3] = pow(x, 2)*(pow(x, 2)*(x*(x*((623.0L/144.0L)*x - 727.0L/48.0L) + 2569.0L/144.0L) - 959.0L/144.0L) - 49.0L/36.0L) + 1;
        poly_val[4] = x*(x*(x*(x*(x*(x*(-623.0L/144.0L*x + 545.0L/36.0L) - 1283.0L/72.0L) + 61.0L/9.0L) - 13.0L/48.0L) + 3.0L/4.0L) + 3.0L/4.0L);
        poly_val[5] = x*(x*(x*(x*(x*(x*((623.0L/240.0L)*x - 2179.0L/240.0L) + 171.0L/16.0L) - 199.0L/48.0L) + 1.0L/6.0L) - 3.0L/40.0L) - 3.0L/20.0L);
        poly_val[6] = x*(x*(x*(x*(x*(x*(-623.0L/720.0L*x + 121.0L/40.0L) - 1283.0L/360.0L) + 101.0L/72.0L) - 1.0L/48.0L) + 1.0L/180.0L) + 1.0L/60.0L);
        poly_val[7] = pow(x, 4)*(x*(x*((89.0L/720.0L)*x - 311.0L/720.0L) + 367.0L/720.0L) - 29.0L/144.0L);
        break;
    case 1:
        poly_val[0] = x*(x*(x*(x*(x*(-623.0L/720.0L*x + 13.0L/5.0L) - 185.0L/72.0L) + 7.0L/9.0L) + 1.0L/16.0L) + 1.0L/90.0L) - 1.0L/60.0L;
        poly_val[1] = x*(x*(x*(x*(x*((4361.0L/720.0L)*x - 2183.0L/120.0L) + 2581.0L/144.0L) - 191.0L/36.0L) - 1.0L/2.0L) - 3.0L/20.0L) + 3.0L/20.0L;
        poly_val[2] = x*(x*(x*(x*(x*(-4361.0L/240.0L*x + 1091.0L/20.0L) - 429.0L/8.0L) + 95.0L/6.0L) + 13.0L/16.0L) + 3.0L/2.0L) - 3.0L/4.0L;
        poly_val[3] = x*(pow(x, 2)*(x*(x*((4361.0L/144.0L)*x - 727.0L/8.0L) + 12845.0L/144.0L) - 959.0L/36.0L) - 49.0L/18.0L);
        poly_val[4] = x*(x*(x*(x*(x*(-4361.0L/144.0L*x + 545.0L/6.0L) - 6415.0L/72.0L) + 244.0L/9.0L) - 13.0L/16.0L) + 3.0L/2.0L) + 3.0L/4.0L;
        poly_val[5] = x*(x*(x*(x*(x*((4361.0L/240.0L)*x - 2179.0L/40.0L) + 855.0L/16.0L) - 199.0L/12.0L) + 1.0L/2.0L) - 3.0L/20.0L) - 3.0L/20.0L;
        poly_val[6] = x*(x*(x*(x*(x*(-4361.0L/720.0L*x + 363.0L/20.0L) - 1283.0L/72.0L) + 101.0L/18.0L) - 1.0L/16.0L) + 1.0L/90.0L) + 1.0L/60.0L;
        poly_val[7] = pow(x, 3)*(x*(x*((623.0L/720.0L)*x - 311.0L/120.0L) + 367.0L/144.0L) - 29.0L/36.0L);
        break;
    case 2:
        poly_val[0] = x*(x*(x*(x*(-623.0L/120.0L*x + 13) - 185.0L/18.0L) + 7.0L/3.0L) + 1.0L/8.0L) + 1.0L/90.0L;
        poly_val[1] = x*(x*(x*(x*((4361.0L/120.0L)*x - 2183.0L/24.0L) + 2581.0L/36.0L) - 191.0L/12.0L) - 1) - 3.0L/20.0L;
        poly_val[2] = x*(x*(x*(x*(-4361.0L/40.0L*x + 1091.0L/4.0L) - 429.0L/2.0L) + 95.0L/2.0L) + 13.0L/8.0L) + 3.0L/2.0L;
        poly_val[3] = pow(x, 2)*(x*(x*((4361.0L/24.0L)*x - 3635.0L/8.0L) + 12845.0L/36.0L) - 959.0L/12.0L) - 49.0L/18.0L;
        poly_val[4] = x*(x*(x*(x*(-4361.0L/24.0L*x + 2725.0L/6.0L) - 6415.0L/18.0L) + 244.0L/3.0L) - 13.0L/8.0L) + 3.0L/2.0L;
        poly_val[5] = x*(x*(x*(x*((4361.0L/40.0L)*x - 2179.0L/8.0L) + 855.0L/4.0L) - 199.0L/4.0L) + 1) - 3.0L/20.0L;
        poly_val[6] = x*(x*(x*(x*(-4361.0L/120.0L*x + 363.0L/4.0L) - 1283.0L/18.0L) + 101.0L/6.0L) - 1.0L/8.0L) + 1.0L/90.0L;
        poly_val[7] = pow(x, 2)*(x*(x*((623.0L/120.0L)*x - 311.0L/24.0L) + 367.0L/36.0L) - 29.0L/12.0L);
        break;
    }
}

void beta_n3_m4(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = x*(x*(x*(x*(x*(x*(x*(x*((29.0L/72.0L)*x - 29.0L/16.0L) + 2231.0L/720.0L) - 859.0L/360.0L) + 25.0L/36.0L) - 1.0L/144.0L) + 1.0L/48.0L) + 1.0L/180.0L) - 1.0L/60.0L);
        poly_val[1] = x*(x*(x*(x*(x*(x*(x*(x*(-203.0L/72.0L*x + 203.0L/16.0L) - 15617.0L/720.0L) + 4009.0L/240.0L) - 3509.0L/720.0L) + 1.0L/12.0L) - 1.0L/6.0L) - 3.0L/40.0L) + 3.0L/20.0L);
        poly_val[2] = x*(x*(x*(x*(x*(x*(x*(x*((203.0L/24.0L)*x - 609.0L/16.0L) + 15617.0L/240.0L) - 3007.0L/60.0L) + 293.0L/20.0L) - 13.0L/48.0L) + 13.0L/48.0L) + 3.0L/4.0L) - 3.0L/4.0L);
        poly_val[3] = pow(x, 2)*(pow(x, 2)*(x*(x*(x*(x*(-1015.0L/72.0L*x + 1015.0L/16.0L) - 15617.0L/144.0L) + 12029.0L/144.0L) - 3521.0L/144.0L) + 7.0L/18.0L) - 49.0L/36.0L) + 1;
        poly_val[4] = x*(x*(x*(x*(x*(x*(x*(x*((1015.0L/72.0L)*x - 1015.0L/16.0L) + 15617.0L/144.0L) - 2005.0L/24.0L) + 881.0L/36.0L) - 13.0L/48.0L) - 13.0L/48.0L) + 3.0L/4.0L) + 3.0L/4.0L);
        poly_val[5] = x*(x*(x*(x*(x*(x*(x*(x*(-203.0L/24.0L*x + 609.0L/16.0L) - 15617.0L/240.0L) + 12031.0L/240.0L) - 235.0L/16.0L) + 1.0L/12.0L) + 1.0L/6.0L) - 3.0L/40.0L) - 3.0L/20.0L);
        poly_val[6] = x*(x*(x*(x*(x*(x*(x*(x*((203.0L/72.0L)*x - 203.0L/16.0L) + 15617.0L/720.0L) - 752.0L/45.0L) + 881.0L/180.0L) - 1.0L/144.0L) - 1.0L/48.0L) + 1.0L/180.0L) + 1.0L/60.0L);
        poly_val[7] = pow(x, 5)*(x*(x*(x*(-29.0L/72.0L*x + 29.0L/16.0L) - 2231.0L/720.0L) + 191.0L/80.0L) - 503.0L/720.0L);
        break;
    case 1:
        poly_val[0] = x*(x*(x*(x*(x*(x*(x*((29.0L/8.0L)*x - 29.0L/2.0L) + 15617.0L/720.0L) - 859.0L/60.0L) + 125.0L/36.0L) - 1.0L/36.0L) + 1.0L/16.0L) + 1.0L/90.0L) - 1.0L/60.0L;
        poly_val[1] = x*(x*(x*(x*(x*(x*(x*(-203.0L/8.0L*x + 203.0L/2.0L) - 109319.0L/720.0L) + 4009.0L/40.0L) - 3509.0L/144.0L) + 1.0L/3.0L) - 1.0L/2.0L) - 3.0L/20.0L) + 3.0L/20.0L;
        poly_val[2] = x*(x*(x*(x*(x*(x*(x*((609.0L/8.0L)*x - 609.0L/2.0L) + 109319.0L/240.0L) - 3007.0L/10.0L) + 293.0L/4.0L) - 13.0L/12.0L) + 13.0L/16.0L) + 3.0L/2.0L) - 3.0L/4.0L;
        poly_val[3] = x*(pow(x, 2)*(x*(x*(x*(x*(-1015.0L/8.0L*x + 1015.0L/2.0L) - 109319.0L/144.0L) + 12029.0L/24.0L) - 17605.0L/144.0L) + 14.0L/9.0L) - 49.0L/18.0L);
        poly_val[4] = x*(x*(x*(x*(x*(x*(x*((1015.0L/8.0L)*x - 1015.0L/2.0L) + 109319.0L/144.0L) - 2005.0L/4.0L) + 4405.0L/36.0L) - 13.0L/12.0L) - 13.0L/16.0L) + 3.0L/2.0L) + 3.0L/4.0L;
        poly_val[5] = x*(x*(x*(x*(x*(x*(x*(-609.0L/8.0L*x + 609.0L/2.0L) - 109319.0L/240.0L) + 12031.0L/40.0L) - 1175.0L/16.0L) + 1.0L/3.0L) + 1.0L/2.0L) - 3.0L/20.0L) - 3.0L/20.0L;
        poly_val[6] = x*(x*(x*(x*(x*(x*(x*((203.0L/8.0L)*x - 203.0L/2.0L) + 109319.0L/720.0L) - 1504.0L/15.0L) + 881.0L/36.0L) - 1.0L/36.0L) - 1.0L/16.0L) + 1.0L/90.0L) + 1.0L/60.0L;
        poly_val[7] = pow(x, 4)*(x*(x*(x*(-29.0L/8.0L*x + 29.0L/2.0L) - 15617.0L/720.0L) + 573.0L/40.0L) - 503.0L/144.0L);
        break;
    case 2:
        poly_val[0] = x*(x*(x*(x*(x*(x*(29*x - 203.0L/2.0L) + 15617.0L/120.0L) - 859.0L/12.0L) + 125.0L/9.0L) - 1.0L/12.0L) + 1.0L/8.0L) + 1.0L/90.0L;
        poly_val[1] = x*(x*(x*(x*(x*(x*(-203*x + 1421.0L/2.0L) - 109319.0L/120.0L) + 4009.0L/8.0L) - 3509.0L/36.0L) + 1) - 1) - 3.0L/20.0L;
        poly_val[2] = x*(x*(x*(x*(x*(x*(609*x - 4263.0L/2.0L) + 109319.0L/40.0L) - 3007.0L/2.0L) + 293) - 13.0L/4.0L) + 13.0L/8.0L) + 3.0L/2.0L;
        poly_val[3] = pow(x, 2)*(x*(x*(x*(x*(-1015*x + 7105.0L/2.0L) - 109319.0L/24.0L) + 60145.0L/24.0L) - 17605.0L/36.0L) + 14.0L/3.0L) - 49.0L/18.0L;
        poly_val[4] = x*(x*(x*(x*(x*(x*(1015*x - 7105.0L/2.0L) + 109319.0L/24.0L) - 10025.0L/4.0L) + 4405.0L/9.0L) - 13.0L/4.0L) - 13.0L/8.0L) + 3.0L/2.0L;
        poly_val[5] = x*(x*(x*(x*(x*(x*(-609*x + 4263.0L/2.0L) - 109319.0L/40.0L) + 12031.0L/8.0L) - 1175.0L/4.0L) + 1) + 1) - 3.0L/20.0L;
        poly_val[6] = x*(x*(x*(x*(x*(x*(203*x - 1421.0L/2.0L) + 109319.0L/120.0L) - 1504.0L/3.0L) + 881.0L/9.0L) - 1.0L/12.0L) - 1.0L/8.0L) + 1.0L/90.0L;
        poly_val[7] = pow(x, 3)*(x*(x*(x*(-29*x + 203.0L/2.0L) - 15617.0L/120.0L) + 573.0L/8.0L) - 503.0L/36.0L);
        break;
    }
}

void beta_n3_m5(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(-503.0L/360.0L*x + 5533.0L/720.0L) - 273.0L/16.0L) + 919.0L/48.0L) - 7829.0L/720.0L) + 601.0L/240.0L) - 1.0L/240.0L) - 1.0L/144.0L) + 1.0L/48.0L) + 1.0L/180.0L) - 1.0L/60.0L);
        poly_val[1] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((3521.0L/360.0L)*x - 38731.0L/720.0L) + 1911.0L/16.0L) - 6433.0L/48.0L) + 54803.0L/720.0L) - 631.0L/36.0L) + 1.0L/60.0L) + 1.0L/12.0L) - 1.0L/6.0L) - 3.0L/40.0L) + 3.0L/20.0L);
        poly_val[2] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(-3521.0L/120.0L*x + 38731.0L/240.0L) - 5733.0L/16.0L) + 6433.0L/16.0L) - 54803.0L/240.0L) + 12619.0L/240.0L) - 1.0L/48.0L) - 13.0L/48.0L) + 13.0L/48.0L) + 3.0L/4.0L) - 3.0L/4.0L);
        poly_val[3] = pow(x, 2)*(pow(x, 2)*(pow(x, 2)*(x*(x*(x*(x*((3521.0L/72.0L)*x - 38731.0L/144.0L) + 9555.0L/16.0L) - 32165.0L/48.0L) + 54803.0L/144.0L) - 701.0L/8.0L) + 7.0L/18.0L) - 49.0L/36.0L) + 1;
        poly_val[4] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(-3521.0L/72.0L*x + 38731.0L/144.0L) - 9555.0L/16.0L) + 32165.0L/48.0L) - 54803.0L/144.0L) + 12617.0L/144.0L) + 1.0L/48.0L) - 13.0L/48.0L) - 13.0L/48.0L) + 3.0L/4.0L) + 3.0L/4.0L);
        poly_val[5] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((3521.0L/120.0L)*x - 38731.0L/240.0L) + 5733.0L/16.0L) - 6433.0L/16.0L) + 54803.0L/240.0L) - 1577.0L/30.0L) - 1.0L/60.0L) + 1.0L/12.0L) + 1.0L/6.0L) - 3.0L/40.0L) - 3.0L/20.0L);
        poly_val[6] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(-3521.0L/360.0L*x + 38731.0L/720.0L) - 1911.0L/16.0L) + 6433.0L/48.0L) - 54803.0L/720.0L) + 841.0L/48.0L) + 1.0L/240.0L) - 1.0L/144.0L) - 1.0L/48.0L) + 1.0L/180.0L) + 1.0L/60.0L);
        poly_val[7] = pow(x, 6)*(x*(x*(x*(x*((503.0L/360.0L)*x - 5533.0L/720.0L) + 273.0L/16.0L) - 919.0L/48.0L) + 7829.0L/720.0L) - 901.0L/360.0L);
        break;
    case 1:
        poly_val[0] = x*(x*(x*(x*(x*(x*(x*(x*(x*(-5533.0L/360.0L*x + 5533.0L/72.0L) - 2457.0L/16.0L) + 919.0L/6.0L) - 54803.0L/720.0L) + 601.0L/40.0L) - 1.0L/48.0L) - 1.0L/36.0L) + 1.0L/16.0L) + 1.0L/90.0L) - 1.0L/60.0L;
        poly_val[1] = x*(x*(x*(x*(x*(x*(x*(x*(x*((38731.0L/360.0L)*x - 38731.0L/72.0L) + 17199.0L/16.0L) - 6433.0L/6.0L) + 383621.0L/720.0L) - 631.0L/6.0L) + 1.0L/12.0L) + 1.0L/3.0L) - 1.0L/2.0L) - 3.0L/20.0L) + 3.0L/20.0L;
        poly_val[2] = x*(x*(x*(x*(x*(x*(x*(x*(x*(-38731.0L/120.0L*x + 38731.0L/24.0L) - 51597.0L/16.0L) + 6433.0L/2.0L) - 383621.0L/240.0L) + 12619.0L/40.0L) - 5.0L/48.0L) - 13.0L/12.0L) + 13.0L/16.0L) + 3.0L/2.0L) - 3.0L/4.0L;
        poly_val[3] = x*(pow(x, 2)*(pow(x, 2)*(x*(x*(x*(x*((38731.0L/72.0L)*x - 193655.0L/72.0L) + 85995.0L/16.0L) - 32165.0L/6.0L) + 383621.0L/144.0L) - 2103.0L/4.0L) + 14.0L/9.0L) - 49.0L/18.0L);
        poly_val[4] = x*(x*(x*(x*(x*(x*(x*(x*(x*(-38731.0L/72.0L*x + 193655.0L/72.0L) - 85995.0L/16.0L) + 32165.0L/6.0L) - 383621.0L/144.0L) + 12617.0L/24.0L) + 5.0L/48.0L) - 13.0L/12.0L) - 13.0L/16.0L) + 3.0L/2.0L) + 3.0L/4.0L;
        poly_val[5] = x*(x*(x*(x*(x*(x*(x*(x*(x*((38731.0L/120.0L)*x - 38731.0L/24.0L) + 51597.0L/16.0L) - 6433.0L/2.0L) + 383621.0L/240.0L) - 1577.0L/5.0L) - 1.0L/12.0L) + 1.0L/3.0L) + 1.0L/2.0L) - 3.0L/20.0L) - 3.0L/20.0L;
        poly_val[6] = x*(x*(x*(x*(x*(x*(x*(x*(x*(-38731.0L/360.0L*x + 38731.0L/72.0L) - 17199.0L/16.0L) + 6433.0L/6.0L) - 383621.0L/720.0L) + 841.0L/8.0L) + 1.0L/48.0L) - 1.0L/36.0L) - 1.0L/16.0L) + 1.0L/90.0L) + 1.0L/60.0L;
        poly_val[7] = pow(x, 5)*(x*(x*(x*(x*((5533.0L/360.0L)*x - 5533.0L/72.0L) + 2457.0L/16.0L) - 919.0L/6.0L) + 54803.0L/720.0L) - 901.0L/60.0L);
        break;
    case 2:
        poly_val[0] = x*(x*(x*(x*(x*(x*(x*(x*(-5533.0L/36.0L*x + 5533.0L/8.0L) - 2457.0L/2.0L) + 6433.0L/6.0L) - 54803.0L/120.0L) + 601.0L/8.0L) - 1.0L/12.0L) - 1.0L/12.0L) + 1.0L/8.0L) + 1.0L/90.0L;
        poly_val[1] = x*(x*(x*(x*(x*(x*(x*(x*((38731.0L/36.0L)*x - 38731.0L/8.0L) + 17199.0L/2.0L) - 45031.0L/6.0L) + 383621.0L/120.0L) - 3155.0L/6.0L) + 1.0L/3.0L) + 1) - 1) - 3.0L/20.0L;
        poly_val[2] = x*(x*(x*(x*(x*(x*(x*(x*(-38731.0L/12.0L*x + 116193.0L/8.0L) - 51597.0L/2.0L) + 45031.0L/2.0L) - 383621.0L/40.0L) + 12619.0L/8.0L) - 5.0L/12.0L) - 13.0L/4.0L) + 13.0L/8.0L) + 3.0L/2.0L;
        poly_val[3] = pow(x, 2)*(pow(x, 2)*(x*(x*(x*(x*((193655.0L/36.0L)*x - 193655.0L/8.0L) + 85995.0L/2.0L) - 225155.0L/6.0L) + 383621.0L/24.0L) - 10515.0L/4.0L) + 14.0L/3.0L) - 49.0L/18.0L;
        poly_val[4] = x*(x*(x*(x*(x*(x*(x*(x*(-193655.0L/36.0L*x + 193655.0L/8.0L) - 85995.0L/2.0L) + 225155.0L/6.0L) - 383621.0L/24.0L) + 63085.0L/24.0L) + 5.0L/12.0L) - 13.0L/4.0L) - 13.0L/8.0L) + 3.0L/2.0L;
        poly_val[5] = x*(x*(x*(x*(x*(x*(x*(x*((38731.0L/12.0L)*x - 116193.0L/8.0L) + 51597.0L/2.0L) - 45031.0L/2.0L) + 383621.0L/40.0L) - 1577) - 1.0L/3.0L) + 1) + 1) - 3.0L/20.0L;
        poly_val[6] = x*(x*(x*(x*(x*(x*(x*(x*(-38731.0L/36.0L*x + 38731.0L/8.0L) - 17199.0L/2.0L) + 45031.0L/6.0L) - 383621.0L/120.0L) + 4205.0L/8.0L) + 1.0L/12.0L) - 1.0L/12.0L) - 1.0L/8.0L) + 1.0L/90.0L;
        poly_val[7] = pow(x, 4)*(x*(x*(x*(x*((5533.0L/36.0L)*x - 5533.0L/8.0L) + 2457.0L/2.0L) - 6433.0L/6.0L) + 54803.0L/120.0L) - 901.0L/12.0L);
        break;
    }
}

void beta_n3_m6(int deriv, double x, double *poly_val)
{
    switch(deriv)
    {
    case 0:
        poly_val[0] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((901.0L/180.0L)*x - 11713.0L/360.0L) + 31933.0L/360.0L) - 93577.0L/720.0L) + 15563.0L/144.0L) - 11623.0L/240.0L) + 6587.0L/720.0L) + 1.0L/720.0L) - 1.0L/240.0L) - 1.0L/144.0L) + 1.0L/48.0L) + 1.0L/180.0L) - 1.0L/60.0L);
        poly_val[1] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(-6307.0L/180.0L*x + 81991.0L/360.0L) - 223531.0L/360.0L) + 655039.0L/720.0L) - 108941.0L/144.0L) + 81361.0L/240.0L) - 46109.0L/720.0L) - 1.0L/120.0L) + 1.0L/60.0L) + 1.0L/12.0L) - 1.0L/6.0L) - 3.0L/40.0L) + 3.0L/20.0L);
        poly_val[2] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((6307.0L/60.0L)*x - 81991.0L/120.0L) + 223531.0L/120.0L) - 655039.0L/240.0L) + 108941.0L/48.0L) - 81361.0L/80.0L) + 46109.0L/240.0L) + 1.0L/48.0L) - 1.0L/48.0L) - 13.0L/48.0L) + 13.0L/48.0L) + 3.0L/4.0L) - 3.0L/4.0L);
        poly_val[3] = pow(x, 2)*(pow(x, 2)*(pow(x, 2)*(x*(x*(x*(x*(x*(x*(-6307.0L/36.0L*x + 81991.0L/72.0L) - 223531.0L/72.0L) + 655039.0L/144.0L) - 544705.0L/144.0L) + 81361.0L/48.0L) - 46109.0L/144.0L) - 1.0L/36.0L) + 7.0L/18.0L) - 49.0L/36.0L) + 1;
        poly_val[4] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((6307.0L/36.0L)*x - 81991.0L/72.0L) + 223531.0L/72.0L) - 655039.0L/144.0L) + 544705.0L/144.0L) - 81361.0L/48.0L) + 46109.0L/144.0L) + 1.0L/48.0L) + 1.0L/48.0L) - 13.0L/48.0L) - 13.0L/48.0L) + 3.0L/4.0L) + 3.0L/4.0L);
        poly_val[5] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(-6307.0L/60.0L*x + 81991.0L/120.0L) - 223531.0L/120.0L) + 655039.0L/240.0L) - 108941.0L/48.0L) + 81361.0L/80.0L) - 46109.0L/240.0L) - 1.0L/120.0L) - 1.0L/60.0L) + 1.0L/12.0L) + 1.0L/6.0L) - 3.0L/40.0L) - 3.0L/20.0L);
        poly_val[6] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((6307.0L/180.0L)*x - 81991.0L/360.0L) + 223531.0L/360.0L) - 655039.0L/720.0L) + 108941.0L/144.0L) - 81361.0L/240.0L) + 46109.0L/720.0L) + 1.0L/720.0L) + 1.0L/240.0L) - 1.0L/144.0L) - 1.0L/48.0L) + 1.0L/180.0L) + 1.0L/60.0L);
        poly_val[7] = pow(x, 7)*(x*(x*(x*(x*(x*(-901.0L/180.0L*x + 11713.0L/360.0L) - 31933.0L/360.0L) + 93577.0L/720.0L) - 15563.0L/144.0L) + 11623.0L/240.0L) - 6587.0L/720.0L);
        break;
    case 1:
        poly_val[0] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((11713.0L/180.0L)*x - 11713.0L/30.0L) + 351263.0L/360.0L) - 93577.0L/72.0L) + 15563.0L/16.0L) - 11623.0L/30.0L) + 46109.0L/720.0L) + 1.0L/120.0L) - 1.0L/48.0L) - 1.0L/36.0L) + 1.0L/16.0L) + 1.0L/90.0L) - 1.0L/60.0L;
        poly_val[1] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(-81991.0L/180.0L*x + 81991.0L/30.0L) - 2458841.0L/360.0L) + 655039.0L/72.0L) - 108941.0L/16.0L) + 81361.0L/30.0L) - 322763.0L/720.0L) - 1.0L/20.0L) + 1.0L/12.0L) + 1.0L/3.0L) - 1.0L/2.0L) - 3.0L/20.0L) + 3.0L/20.0L;
        poly_val[2] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((81991.0L/60.0L)*x - 81991.0L/10.0L) + 2458841.0L/120.0L) - 655039.0L/24.0L) + 326823.0L/16.0L) - 81361.0L/10.0L) + 322763.0L/240.0L) + 1.0L/8.0L) - 5.0L/48.0L) - 13.0L/12.0L) + 13.0L/16.0L) + 3.0L/2.0L) - 3.0L/4.0L;
        poly_val[3] = x*(pow(x, 2)*(pow(x, 2)*(x*(x*(x*(x*(x*(x*(-81991.0L/36.0L*x + 81991.0L/6.0L) - 2458841.0L/72.0L) + 3275195.0L/72.0L) - 544705.0L/16.0L) + 81361.0L/6.0L) - 322763.0L/144.0L) - 1.0L/6.0L) + 14.0L/9.0L) - 49.0L/18.0L);
        poly_val[4] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((81991.0L/36.0L)*x - 81991.0L/6.0L) + 2458841.0L/72.0L) - 3275195.0L/72.0L) + 544705.0L/16.0L) - 81361.0L/6.0L) + 322763.0L/144.0L) + 1.0L/8.0L) + 5.0L/48.0L) - 13.0L/12.0L) - 13.0L/16.0L) + 3.0L/2.0L) + 3.0L/4.0L;
        poly_val[5] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(-81991.0L/60.0L*x + 81991.0L/10.0L) - 2458841.0L/120.0L) + 655039.0L/24.0L) - 326823.0L/16.0L) + 81361.0L/10.0L) - 322763.0L/240.0L) - 1.0L/20.0L) - 1.0L/12.0L) + 1.0L/3.0L) + 1.0L/2.0L) - 3.0L/20.0L) - 3.0L/20.0L;
        poly_val[6] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((81991.0L/180.0L)*x - 81991.0L/30.0L) + 2458841.0L/360.0L) - 655039.0L/72.0L) + 108941.0L/16.0L) - 81361.0L/30.0L) + 322763.0L/720.0L) + 1.0L/120.0L) + 1.0L/48.0L) - 1.0L/36.0L) - 1.0L/16.0L) + 1.0L/90.0L) + 1.0L/60.0L;
        poly_val[7] = pow(x, 6)*(x*(x*(x*(x*(x*(-11713.0L/180.0L*x + 11713.0L/30.0L) - 351263.0L/360.0L) + 93577.0L/72.0L) - 15563.0L/16.0L) + 11623.0L/30.0L) - 46109.0L/720.0L);
        break;
    case 2:
        poly_val[0] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((11713.0L/15.0L)*x - 128843.0L/30.0L) + 351263.0L/36.0L) - 93577.0L/8.0L) + 15563.0L/2.0L) - 81361.0L/30.0L) + 46109.0L/120.0L) + 1.0L/24.0L) - 1.0L/12.0L) - 1.0L/12.0L) + 1.0L/8.0L) + 1.0L/90.0L;
        poly_val[1] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(-81991.0L/15.0L*x + 901901.0L/30.0L) - 2458841.0L/36.0L) + 655039.0L/8.0L) - 108941.0L/2.0L) + 569527.0L/30.0L) - 322763.0L/120.0L) - 1.0L/4.0L) + 1.0L/3.0L) + 1) - 1) - 3.0L/20.0L;
        poly_val[2] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((81991.0L/5.0L)*x - 901901.0L/10.0L) + 2458841.0L/12.0L) - 1965117.0L/8.0L) + 326823.0L/2.0L) - 569527.0L/10.0L) + 322763.0L/40.0L) + 5.0L/8.0L) - 5.0L/12.0L) - 13.0L/4.0L) + 13.0L/8.0L) + 3.0L/2.0L;
        poly_val[3] = pow(x, 2)*(pow(x, 2)*(x*(x*(x*(x*(x*(x*(-81991.0L/3.0L*x + 901901.0L/6.0L) - 12294205.0L/36.0L) + 3275195.0L/8.0L) - 544705.0L/2.0L) + 569527.0L/6.0L) - 322763.0L/24.0L) - 5.0L/6.0L) + 14.0L/3.0L) - 49.0L/18.0L;
        poly_val[4] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((81991.0L/3.0L)*x - 901901.0L/6.0L) + 12294205.0L/36.0L) - 3275195.0L/8.0L) + 544705.0L/2.0L) - 569527.0L/6.0L) + 322763.0L/24.0L) + 5.0L/8.0L) + 5.0L/12.0L) - 13.0L/4.0L) - 13.0L/8.0L) + 3.0L/2.0L;
        poly_val[5] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(-81991.0L/5.0L*x + 901901.0L/10.0L) - 2458841.0L/12.0L) + 1965117.0L/8.0L) - 326823.0L/2.0L) + 569527.0L/10.0L) - 322763.0L/40.0L) - 1.0L/4.0L) - 1.0L/3.0L) + 1) + 1) - 3.0L/20.0L;
        poly_val[6] = x*(x*(x*(x*(x*(x*(x*(x*(x*(x*((81991.0L/15.0L)*x - 901901.0L/30.0L) + 2458841.0L/36.0L) - 655039.0L/8.0L) + 108941.0L/2.0L) - 569527.0L/30.0L) + 322763.0L/120.0L) + 1.0L/24.0L) + 1.0L/12.0L) - 1.0L/12.0L) - 1.0L/8.0L) + 1.0L/90.0L;
        poly_val[7] = pow(x, 5)*(x*(x*(x*(x*(x*(-11713.0L/15.0L*x + 128843.0L/30.0L) - 351263.0L/36.0L) + 93577.0L/8.0L) - 15563.0L/2.0L) + 81361.0L/30.0L) - 46109.0L/120.0L);
        break;
    }
}

