import numpy as np

def count_expensive(fk0, fk1):
    kcomponent = np.arange(-np.floor(fk1)-1, np.floor(fk1)+2, 1).astype(np.float)
    ksize = (kcomponent[:, None, None]**2 +
             kcomponent[None, :, None]**2 +
             kcomponent[None, None, :]**2)**.5
    #print(ksize[0])

    good_indices = np.where(np.logical_and(
        ksize >= fk0,
        ksize <= fk1))
    #print(ksize[good_indices])
    #print(good_indices[0].shape)
    return np.unique(ksize[good_indices].flatten(), return_counts = True)

def main():
    for ff in [[1, 2],
               [1.4, 2.3],
               [1.4, 2.2]]:
        modes, counts = count_expensive(ff[0], ff[1])
        nmodes = np.sum(counts)
        print(1 / ff[1], ff, nmodes)
        modes_str  = ''
        counts_str = ''
        for ii in range(counts.shape[0]):
            modes_str += '{0:>5g}\t'.format(modes[ii])
            counts_str += '{0:>5g}\t'.format(counts[ii])
        print(modes_str + '\n' + counts_str + '\n')
    return None

if __name__ == '__main__':
    main()

