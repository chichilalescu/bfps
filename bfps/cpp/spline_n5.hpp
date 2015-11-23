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



#ifndef SPLINE_N5

#define SPLINE_N5

void beta_n5_m0( const int deriv, const double x, double *__restrict__ poly_val);
void beta_n5_m1( const int deriv, const double x, double *__restrict__ poly_val);
void beta_n5_m2( const int deriv, const double x, double *__restrict__ poly_val);
void beta_n5_m3( const int deriv, const double x, double *__restrict__ poly_val);
void beta_n5_m4( const int deriv, const double x, double *__restrict__ poly_val);
void beta_n5_m5( const int deriv, const double x, double *__restrict__ poly_val);
void beta_n5_m6( const int deriv, const double x, double *__restrict__ poly_val);
void beta_n5_m7( const int deriv, const double x, double *__restrict__ poly_val);
void beta_n5_m8( const int deriv, const double x, double *__restrict__ poly_val);
void beta_n5_m9( const int deriv, const double x, double *__restrict__ poly_val);
void beta_n5_m10(const int deriv, const double x, double *__restrict__ poly_val);

#endif//SPLINE_N5

