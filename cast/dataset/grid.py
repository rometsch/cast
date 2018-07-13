#----------------------------------------------------------------------
#   Grid structure to represent different geometries
#
#	Author 	: Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#	Date	: 2018-07-13
#----------------------------------------------------------------------

import numpy as np
import sys

class AbstractGrid:
    def __init__(self):
        self._shape()
        self._dx()

    def __repr__(self):
        classname = repr(self.__class__)[8:-2]
        rv = "<" + classname + ": {}-d, ({}) = {}>".format(
            self.dim, ",".join(["N{}".format(l) for l in self.dim_labels]), self.shape )
        return rv

    def __getitem__(self, key):
        try:
            return self.__dict__[key]
        except KeyError:
            try:
                getattr(self, "_{}".format(key))()
                return self.__dict__[key]
            except AttributeError:
                raise AttributeError("{} object has no attribute '{}' and no function to calculate it".format(type(self), key))

    def _shape(self):
        self.shape = tuple(len(self.__dict__[l]) for l in self.dim_labels)
        for l,N in zip(self.dim_labels, self.shape):
            self.__dict__["N" + l] = N

    def _dx(self):
        for l in self.dim_labels:
            if not "d" + l in self.__dict__:
                x = self.__dict__[l]
                dx = x[1:] - x[:-1]
                if all(np.isclose(dx, dx[0])):
                    self.__dict__["d" + l] = dx[0]
                else:
                    raise ValueError("Unequal spacing found in {} but d{} is not specified.".format(l,l))

    def _dV(self):
        for l in self.dim_labels:
            pass


class SphericalRegularGrid(AbstractGrid):
    def __init__(self, r=None, theta=None, phi=None, dr=None, dphi=None, dtheta=None):
        self.dim = dim = sum(x is not None for x in [r, theta, phi])

        if r is None:
            raise ValueError("Can't have a spherical grid without radius. r={}".format(dim,r))
        self.dim_labels = ["r"]
        self.r = r

        if dim==2:
            if theta is not None:
                self.theta = theta
                self.dim_labels.append("theta")
            elif phi is not None:
                self.phi = phi
                self.dim_labels.append("phi")

        elif dim==3:
            self.theta = theta
            self.phi = phi
            self.dim_labels += ["theta", "phi"]

        if dr is not None:
            self.dr = dr
        if dtheta is not None:
            self.dtheta = dtheta
        if dphi is not None:
            self.dphi = dphi

        super().__init__()

    def _dVr(self):
        r = self.r
        dr = self.dr
        self.dVr = ( (r+dr/2)**3 - (r-dr/2)**3 )/3


    def _dVtheta(self):
        th = self.theta
        dth = self.dtheta
        self.dVtheta = np.cos( th - dth/2 ) - np.cos( th + dth/2 )


    def _dVphi(self):
        try:
            if len(self.dphi) == self.Nphi:
                self.dVphi = self.dphi
            else:
                raise ValueError("dphi has len = {} instead of {}".format(len(self.dphi), self.Nphi))
        except TypeError:
            self.dVphi = np.ones(self.Nphi)*self.dphi
