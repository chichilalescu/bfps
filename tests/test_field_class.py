import numpy as np
import matplotlib.pyplot as plt
import pyfftw
import bfps
import bfps.tools

def main():
    kdata = pyfftw.n_byte_align_empty(
            (32, 32, 17),
            pyfftw.simd_alignment,
            dtype = np.complex64)
    rdata = pyfftw.n_byte_align_empty(
            (32, 32, 32),
            pyfftw.simd_alignment,
            dtype = np.float32)
    c2r = pyfftw.FFTW(
            kdata.transpose((1, 0, 2)),
            rdata,
            axes = (0, 1, 2),
            direction = 'FFTW_BACKWARD',
            threads = 8)
    kdata[:] = bfps.tools.generate_data_3D(32, 32, 32, dtype = np.complex64)
    c2r.execute()
    fig = plt.figure()
    a = fig.add_subplot(111)
    a.set_axis_off()
    a.imshow(rdata[:, 4, :], interpolation = 'none')
    fig.tight_layout()
    fig.savefig('tst.pdf')
    return None

if __name__ == '__main__':
    main()

