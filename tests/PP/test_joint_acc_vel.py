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
    df = h5py.File('test.h5', 'r')
    vel_hist_regular = df['statistics/histograms/velocity'].value
    df.close()

    f = plt.figure()
    a = f.add_subplot(211)
    a.plot(acc_hist[0, :, :3])
    a.plot(np.sum(acc_vel_histc[0, :, :, :, 0], axis = 1), dashes = (4, 4))
    a = f.add_subplot(212)
    a.plot(acc_hist[0, :, 3])
    a.plot(np.sum(acc_vel_histm[0, :, :], axis = 1), dashes = (4, 4))
    f.tight_layout()
    f.savefig('sanity_test_acceleration.pdf')
    plt.close(f)

    f = plt.figure()
    a = f.add_subplot(211)
    a.plot(vel_hist[0, :, :3])
    a.plot(vel_hist_regular[0, :, :3], dashes = (1, 1))
    a.plot(np.sum(acc_vel_histc[0, :, :, 0, :], axis = 0), dashes = (4, 4))
    a = f.add_subplot(212)
    hh = vel_hist[0, :, 3]
    a.plot(hh)
    print(np.sum(hh))
    hh = np.sum(acc_vel_histm[0, :, :], axis = 0)
    a.plot(hh, dashes = (4, 4))
    print(np.sum(hh))
    hh = vel_hist_regular[0, :, 3]
    a.plot(hh, dashes = (1, 1))
    print(np.sum(hh))
    f.tight_layout()
    f.savefig('sanity_test_velocity.pdf')
    plt.close(f)
    return None

if __name__ == '__main__':
    main()

