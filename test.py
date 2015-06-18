import numpy as np
import subprocess
import pyfftw

def run_test(
        test_name = 'test_FFT',
        ncpu = 4):
    # first, write file
    with open('src/' + test_name + '.cpp', 'w') as outfile:
        for fname in ['py_template/start.cpp',
                      'py_template/' + test_name + '.cpp',
                      'py_template/end.cpp']:
            with open(fname) as infile:
                outfile.write(infile.read() + '\n')

    # now compile code and run
    if subprocess.call(['make', test_name + '.elf']) == 0:
        subprocess.call(['time',
                         'mpirun',
                         '-np',
                         '{0}'.format(ncpu),
                         './' + test_name + '.elf'])
    return None

def generate_data_3D(
        n,
        dtype = np.complex128,
        p = 1.5):
    """
    generate something that has the proper shape
    """
    assert(n % 2 == 0)
    a = np.zeros((n, n, n/2+1), dtype = dtype)
    a[:] = np.random.randn(*a.shape) + 1j*np.random.randn(*a.shape)
    k, j, i = np.mgrid[-n/2+1:n/2+1, -n/2+1:n/2+1, 0:n/2+1]
    k = (k**2 + j**2 + i**2)**.5
    k = np.roll(k, n//2+1, axis = 0)
    k = np.roll(k, n//2+1, axis = 1)
    a /= k**p
    a[0, :, :] = 0
    a[:, 0, :] = 0
    a[:, :, 0] = 0
    ii = np.where(k == 0)
    a[ii] = 0
    ii = np.where(k > n/3)
    a[ii] = 0
    return a

Kdata0 = generate_data_3D(32, p = 2).astype(np.complex64)
Kdata0.tofile("Kdata0")
run_test('test_FFT')
Kdata1 = np.fromfile('Kdata1', dtype = np.complex64).reshape(Kdata0.shape)

print np.max(np.abs(Kdata0 - Kdata1))




