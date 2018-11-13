from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

#cmaps: plasma, rainbow, jet

def compute_density(x, y):
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    return z
    
def density_scatterplot(x, y, z, s=10, size=(6,9)):
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots(figsize=size)
    im = ax.scatter(x, y, c=z, s=s, edgecolor='', cmap=plt.get_cmap("plasma"))
    
    plt.colorbar(im)
    
    return fig, ax