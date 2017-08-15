import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    s1 = np.sum(hh)
    hh = np.sum(acc_vel_histm[0, :, :], axis = 0)
    a.plot(hh, dashes = (4, 4))
    s2 = np.sum(hh)
    hh = vel_hist_regular[0, :, 3]
    a.plot(hh, dashes = (1, 1))
    s3 = np.sum(hh)
    assert(s1 == s2)
    assert(s1 == s3)
    f.tight_layout()
    f.savefig('sanity_test_velocity.pdf')
    plt.close(f)

    f = plt.figure(figsize = (10, 5))
    gs = gridspec.GridSpec(
            3, 6)
    for i in range(3):
        for j in range(3):
            a = f.add_subplot(gs[i, j])
            a.imshow(acc_vel_histc[0, :, :, i, j])
            a.set_axis_off()
    a = f.add_subplot(gs[0:, 3:])
    a.imshow(acc_vel_histm[0])
    a.set_axis_off()
    f.tight_layout()
    f.savefig('joing_acceleration_and_velocity.pdf')
    plt.close(f)
    return None

if __name__ == '__main__':
    main()

