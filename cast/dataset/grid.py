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
        self.dV = self["dV" + self.dim_labels[0]]
        for n in range(1,self.dim):
            l = self.dim_labels[n]
            self.dV = np.repeat( np.expand_dims(self.dV, n), self["N"+l], axis=n)*self["dV"+l]

    def _check_limit(self, name, lim):
        if name in self.__dict__:
            X = self.__dict__[name]
            if not (all(X >= lim[0]) and all(X <= lim[1])):
                raise ValueError(name + " must be in range [{},{}] but min = {}, max = {}".format(
                    lim[0], lim[1], np.min(X), np.max(X)))


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

        if dr is not None:
            self.dr = dr
        if dtheta is not None:
            self.dtheta = dtheta
        if dphi is not None:
            self.dphi = dphi

        elif dim==3:
            self.theta = theta
            self.phi = phi
            self.dim_labels += ["theta", "phi"]

        self._check_limit("r", [0,np.inf])
        self._check_limit("theta", [0,np.pi])
        self._check_limit("phi", [0, 2*np.pi])

        super().__init__()

    def _V(self):
        self.V = np.sum(self["dV"])

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
