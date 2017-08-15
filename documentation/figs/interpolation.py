import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

def main():
    slab = 2
    nproc = 5
    f = plt.figure(figsize = (6, 4.5))
    a = f.add_subplot(111)
    for p in range(nproc):
        color = plt.get_cmap('plasma')(p*1./nproc)
        a.add_patch(
                mpatches.Rectangle(
                        [0, p*slab],
                        slab*(nproc+2)-1, 1,
                        color = color,
                        alpha = .2))
        a.text(-.5, p*slab+.5, '$p_{0}$'.format(p),
               verticalalignment = 'center')
        for y in range((nproc+2)*slab):
            a.plot([y, y],
                   range(p*slab, (p+1)*slab),
                   marker = '.',
                   linestyle = 'none',
                   color = color)
    for X, Y in [(9.9, 6.3),
                 (3.3, 3.7)]:
        a.plot([X], [Y],
               color = 'black',
               marker = 'x')
        for n in [1, 2]:
            a.add_patch(
                    mpatches.Rectangle(
                            [math.floor(X-n), math.floor(Y-n)],
                            2*n+1, 2*n+1,
                            color = 'green',
                            alpha = .2))
            a.text(math.floor(X)+.5, math.floor(Y - n)-.3,
                   '$n = {0}$'.format(n),
                   horizontalalignment = 'center')
    a.set_ylim(bottom = -1, top = 10)
    a.set_xlim(left = -1)
    a.set_ylabel('$z$')
    a.set_xlabel('$x,y$')
    a.set_aspect('equal')
    f.tight_layout()
    f.savefig('interp_problem.pdf')
    return None

if __name__ == '__main__':
    main()

