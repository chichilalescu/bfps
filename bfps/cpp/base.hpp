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


#include <mpi.h>
#include <stdarg.h>
#include <iostream>

#ifndef BASE

#define BASE

static const int message_buffer_length = 1024;
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

