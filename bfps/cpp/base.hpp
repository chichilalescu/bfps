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



#include <mpi.h>
#include <stdarg.h>
#include <iostream>
#include <typeinfo>

#ifndef BASE

#define BASE

static const int message_buffer_length = 2048;
extern int myrank, nprocs;

inline int MOD(int a, int n)
{
    return ((a%n) + n) % n;
}

#ifdef OMPI_MPI_H

#define BFPS_MPICXX_DOUBLE_COMPLEX MPI_DOUBLE_COMPLEX

#else

#define BFPS_MPICXX_DOUBLE_COMPLEX MPI_C_DOUBLE_COMPLEX

#endif//OMPI_MPI_H


#ifndef NDEBUG

static char debug_message_buffer[message_buffer_length];

inline void DEBUG_MSG(const char * format, ...)
{
    va_list argptr;
    va_start(argptr, format);
    sprintf(
            debug_message_buffer,
            "cpu%.4d ",
            myrank);
    vsnprintf(
            debug_message_buffer + 8,
            message_buffer_length - 8,
            format,
            argptr);
    va_end(argptr);
    std::cerr << debug_message_buffer;
}

inline void DEBUG_MSG_WAIT(MPI_Comm communicator, const char * format, ...)
{
    va_list argptr;
    va_start(argptr, format);
    sprintf(
            debug_message_buffer,
            "cpu%.4d ",
            myrank);
    vsnprintf(
            debug_message_buffer + 8,
            message_buffer_length - 8,
            format,
            argptr);
    va_end(argptr);
    std::cerr << debug_message_buffer;
    MPI_Barrier(communicator);
}

#else

#define DEBUG_MSG(...)
#define DEBUG_MSG_WAIT(...)

#endif//NDEBUG

#endif//BASE

