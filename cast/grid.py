#----------------------------------------------------------------------
#   Grid structure to represent different geometries
#
#	Author 	: Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#	Date	: 2018-07-13
#----------------------------------------------------------------------

import numpy as np
import sys

class AbstractGrid:
    def __init__(self, domains_expected = [], limits = [], **kwargs):
        self.dim = 0
        self.dim_labels = []

        for l in ["r", "theta", "phi"]:
            if l in kwargs:
                self.dim += 1
                self.dim_labels.append(l)
                self.__dict__[l] = kwargs[l]
            if "d"+l in kwargs:
                self.__dict__["d"+l] = kwargs["d"+l]

        for l, limit in zip(domains_expected, limits):
            self._check_limit(l, limit)

        self._shape()
        self._dx()
        self._xI()

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

    def _xI(self):
        # Calculate coordinates of cell interfaces
        # Resulting array is one element larger
        for l in self.dim_labels:
            if not l+'I' in self.__dict__:
                x = self.__dict__[l]
                dx = self.__dict__["d"+l]
                try:
                    last_dx = dx[-1]
                except TypeError:
                    last_dx = dx
                xI = np.append( x-dx/2, x[-1] + last_dx/2)
                self.__dict__[l+"I"] = xI

    def _dV(self):
        self.dV = self["dV" + self.dim_labels[0]]
        for n in range(1,self.dim):
            l = self.dim_labels[n]
            try:
                unit = self.dV.unit
            except AttributeError:
                unit = 1
            self.dV = np.repeat( np.expand_dims(self.dV, n), self["N"+l], axis=n)*self["dV"+l]*unit


    def _check_limit(self, name, lim):
        if name in self.__dict__:
            X = self.__dict__[name]
            if not (all(X >= lim[0]) and all(X <= lim[1])):
                raise ValueError(name + " must be in range [{},{}] but min = {}, max = {}".format(
                    lim[0], lim[1], np.min(X), np.max(X)))

    def meshgrid(self):
        self._xI()
        X = [self.__dict__[l+"I"] for l in reversed(self.dim_labels)]
        if self.dim == 1:
            return np.meshgrid( X[0] )
        elif self.dim == 2:
            return np.meshgrid( X[1], X[0] )
        elif self.dim == 3:
            return np.meshgrid( X[2], X[1], X[0] )
        else:
            raise ValueError("Can not construct meshgrid for dim = {}".format(self.dim))

class SphericalRegularGrid(AbstractGrid):
    def __init__(self, **kwargs):
        super().__init__(domains_expected = ["r", "theta", "phi"],
                         limits = [[0, np.inf], [0, np.pi], [0, 2*np.pi]],
                         **kwargs)
        if "r" not in self.__dict__:
            raise ValueError("Can't have a spherical grid without radius.")

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

    def meshgrid(self):
        if self.dim == 1:
            return np.meshgrid( self.rI )
        elif self.dim == 2:
            return self._meshgrid_2d()
        elif self.dim == 3:
            return self._meshgrid_3d()
        else:
            raise ValueError("Can not construct meshgrid for dim = {}".format(self.dim))

    def meshgrid_midplane(self):
        if self.dim != 3:
            raise ValueError("Midplane meshgrid only defined for 3d.")
        return self._meshgrid_2d()

    def _meshgrid_2d(self):
        Phi, R = np.meshgrid( self.phiI, self.rI  )
        X = R*np.cos(Phi)
        Y = R*np.sin(Phi)
        return (X,Y)

    def _meshgrid_3d(self):
        Phi, Theta, R = np.meshgrid( self.phiI, self.phiI , self.rI )
        X = R*np.cos(Phi)*np.sin(Theta)
        Y = R*np.sin(Phi)*np.sin(Theta)
        Z = R*np.cos(Theta)
        return ( X, Y, Z )

class PolarRegularGrid(AbstractGrid):
    def __init__(self, **kwargs):
        super().__init__(domains_expected = ["r", "phi"],
                         limits = [[0, np.inf], [0, 2*np.pi]],
                         **kwargs)
        if "r" not in self.__dict__:
            raise ValueError("Can't have a polar grid without radius.")

    def _V(self):
        self.V = np.sum(self["dV"])

    def _dVr(self):
        r = self.r
        dr = self.dr
        self.dVr = 0.5*( (r+dr/2)**2 - (r-dr/2)**2 )

    def _dVphi(self):
        try:
            if len(self.dphi) == self.Nphi:
                self.dVphi = self.dphi
            else:
                raise ValueError("dphi has len = {} instead of {}".format(len(self.dphi), self.Nphi))
        except TypeError:
            self.dVphi = np.ones(self.Nphi)*self.dphi

    def meshgrid(self):
        if self.dim == 1:
            return np.meshgrid( self.rI )
        elif self.dim == 2:
            Phi, R = np.meshgrid( self.phiI, self.rI )
            X = R*np.cos(Phi)
            Y = R*np.sin(Phi)
            return (X,Y)
        else:
            raise ValueError("Can not construct meshgrid for dim = {}".format(self.dim))
