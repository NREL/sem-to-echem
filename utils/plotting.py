import matplotlib.pyplot as plt

def plot_volume(volume, title=None, ax=None, show_zero=False, cmap='viridis'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if show_zero:
        volume += 1
    z, x, y = volume.nonzero()
    c = volume[volume.nonzero()]
    ax.scatter(x, y, z, c=c, alpha=0.5, cmap=cmap)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])
    ax.set_axis_off()
    ax.set_title(title)
    if ax is None:
        plt.show()