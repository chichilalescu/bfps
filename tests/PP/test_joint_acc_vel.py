import numpy as np
import h5py
import matplotlib.pyplot as plt

def main():
    df = h5py.File('test_post.h5', 'r')
    acc_hist = df['joint_acc_vel_stats/histograms/acceleration'].value
    vel_hist = df['joint_acc_vel_stats/histograms/velocity'].value
    acc_vel_histc = df['joint_acc_vel_stats/histograms/acceleration_and_velocity_components'].value
    acc_vel_histm = df['joint_acc_vel_stats/histograms/acceleration_and_velocity_magnitudes'].value
    df.close()

    f = plt.figure()
    a = f.add_subplot(111)
    a.plot(acc_hist[0, :, :3])
    a.plot(np.sum(acc_vel_histc[0, :, :, :, 0], axis = 1), dashes = (1, 1))
    f.savefig('sanity_test.pdf')
    plt.close(f)
    return None

if __name__ == '__main__':
    main()

