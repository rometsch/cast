#----------------------------------------------------------------------
#	Plotting data from cast datasets.
#
#	Author	: Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#	Date	: 2018-08-20
#----------------------------------------------------------------------

from . import grid
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
quantity_support()

def plot_disk_midplane_density(fld, grd=None, ax=None, **kwargs):
    if ax is None:
        fig, myax = plt.subplots()
    else:
        myax = ax

    if grd is None:
        grd = fld.grid
    else:
        grd = grd

    if fld.grid.dim == 3:
        im =_plot_disk_midplane_density_3d(myax, fld, grd, **kwargs)

    if ax is None:
        fig.colorbar(im)
        plt.show()

    return im

def plot_disk_midplane_density_delta(fld, fld0, grd=None, ax=None, **kwargs):
    # same as plot_disk_midplane_density but with a new field holding the difference of data = fld - fld0
    FieldClass = type(fld)
    diff_fld = FieldClass( name=fld.name+"-diff", grid=fld.grid, data = fld.data - fld0.data)
    return plot_disk_midplane_density(diff_fld, grd=grd , ax=ax  , **kwargs)

def _plot_disk_midplane_density_3d(ax, fld, grd, **kwargs):
    # assume a full disk with theta = pi/2 +- delta

    if not isinstance(grd, grid.SphericalRegularGrid):
        raise TypeError("Don't know how to plot midplane density for grids other than SphericalRegularGrid")

    Nmid = int(grd.Ntheta/2)
    if grd.Ntheta%2 == 0:
        data = 0.5*(fld.data[:,Nmid,:] + fld.data[:,Nmid-1,:])
    else:
        data = fld.data[:,Nmid,:]

    X, Y = grd.meshgrid_midplane()

    return ax.pcolormesh(X, Y, data)
