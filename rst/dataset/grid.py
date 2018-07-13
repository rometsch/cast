#----------------------------------------------------------------------
#   Grid structure to represent different geometries
#
#	Author 	: Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#	Date	: 2018-07-13
#----------------------------------------------------------------------

import numpy as np

class AbstractGrid:
    def __init__(self, dim):
        self.dim = dim



class SphericalRegularGrid():
    def __init__(self, dim, r=None, theta=None, phi=None):
        super().__init__(dim)

        if not r:
            raise ValueError("Can't have a {}-d spherical grid without radius. r={}".format(dim,r))
        self.dim_labels = ["r"]

        if dim==1:
            self.r = r

        elif dim==2:
            if (theta and phi):
                raise ValueError("Can't have a 2-d spherical grid with both theta={} and phi={}".format(theta, phi))
            if theta:
                self.theta = theta
                self.dim_labels.append("theta")
            elif phi:
                self.phi = phi
                self.dim_labels.append("phi")
            else:
                raise ValueError("Can't have a 2-d spherical grid without specifying theta={} or phi={}.".format(theta, phi))

        elif dim==3:
            if not (theta and phi):
                raise ValueError("Can't have a 3-d spherical grid without specifying theta={} and phi={}.".format(theta, phi))
            self.theta = theta
            self.phi = phi

        self.__shape()
        self.__dx()

    def __shape(self):
        for l in self.dim_labels:
            self.shape = tuple(len(self.__dict__[l]) for l in self.dim_labels)

    def __dx(self):
        for l in self.dim_labels:
            x = self.__dict__[l]
            self.__dict__["d" + l] = x[1:] - x[:-1]
