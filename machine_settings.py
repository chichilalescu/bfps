import os

########################################################################
#### these you're supposed to adapt to your environment
########################################################################

hostname = os.getenv('HOSTNAME')

extra_compile_args = ['-Wall', '-O2', '-g', '-mtune=native', '-ffast-math', '-std=c++11']
#extra_compile_args = ['-Wall', '-O0', '-g', '-std=c++11']
extra_libraries = ['hdf5_cpp', 'hdf5']

if hostname == 'chichi-G':
    include_dirs = ['/usr/local/include',
                    '/usr/include/mpich']
    library_dirs = ['/usr/local/lib'
                    '/usr/lib/mpich']
    extra_libraries += ['mpich']

if hostname in ['frontend01', 'frontend02']:
    include_dirs = ['/usr/nld/mvapich2-1.9a2-gcc/include',
                    '/usr/nld/gcc-4.7.2/include',
                    '/usr/nld/hdf5-1.8.9/include',
                    '/usr/nld/fftw-3.3.3-mvapich2-1.9a2-gcc/include',
                    '/usr/nld/fftw-3.3.3-float-mvapich2-1.9a2-gcc/include']

    library_dirs = ['/usr/nld/mvapich2-1.9a2-gcc/lib',
                    '/usr/nld/gcc-4.7.2/lib64',
                    '/usr/nld/hdf5-1.8.9/lib',
                    '/usr/nld/fftw-3.3.3-mvapich2-1.9a2-gcc/lib',
                    '/usr/nld/fftw-3.3.3-float-mvapich2-1.9a2-gcc/lib']
    extra_libraries += ['mpich']

if hostname in ['tolima', 'misti']:
    local_install_dir = '/scratch.local/chichi/installs'

    include_dirs = ['/usr/lib64/mpi/gcc/openmpi/include',
                    os.path.join(local_install_dir, 'include')]

    library_dirs = ['/usr/lib64/mpi/gcc/openmpi/lib64',
                    os.path.join(local_install_dir, 'lib'),
                    os.path.join(local_install_dir, 'lib64')]
    extra_libraries += ['mpi_cxx', 'mpi']

