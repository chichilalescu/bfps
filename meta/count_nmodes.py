import numpy as np

def count_expensive(fk0, fk1):
    kcomponent = np.arange(-fk1-1, fk1+2, 1).astype(np.float)
    ksize = (kcomponent[:, None, None]**2 +
             kcomponent[None, :, None]**2 +
             kcomponent[None, None, :]**2)**.5

    good_indices = np.where(np.logical_and(
        ksize >= fk0,
        ksize <= fk1))
    #print(ksize[good_indices])
    #print(good_indices[0].shape)
    return good_indices[0].shape[0]

def main():
    for ff in [[2, 4],
               [1.5, 3],
               [1, 2],
               [1.5, 2.5],
               [1.5, 2.3]]:
        print(1 / ff[1], ff, count_expensive(ff[0], ff[1]))
    return None

if __name__ == '__main__':
    main()

