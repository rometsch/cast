#----------------------------------------------------------------------
#   This file provides a generic dataset class
#
#   Author  : Thomas Rometsch (thomas.rometsch@student.uni-tuebingen.de)
#   Date    : 2017-12-13
#----------------------------------------------------------------------

import numpy as np
from collections import OrderedDict as ODict
from .getter import Getter
from .particle import *
import os

def find_dir_containing(patterns, rootdir):
    if isinstance(patterns, str):
        patterns = [patterns]
    # Look for an output dir inside the rootdir
    for dirname, dirnames, filenames in os.walk(rootdir):
        for subdirname in dirnames + ['']:
            path = os.path.join(dirname, subdirname)
            if all(s in os.listdir(path) for s in patterns):
                return path
    raise FileNotFoundError("Could not find a directory with files matching the patterns '{}' in the (sub)directories of '{}'".format(patterns, rootdir))

code_identifiers = {
    'fargoTwam' : 'misc.dat',
    'fargo3d' : 'fargo3d.h',
    'pluto' : 'pluto.log'
    }

def Dataset(root='.'):
    # A wrapper to determine the correct class
    code = None
    for dirname, dirnames, filenames in os.walk(root):
        for subdirname in dirnames + ['']:
            for key in code_identifiers:
                if code_identifiers[key] in filenames:
                    code = key
    if code == 'fargoTwam':
        from .fargoTwam import FargoTwamDataset as DSet
    elif code == "fargo3d":
        from .fargo3d import Fargo3dDataset as DSet
    elif code == "pluto":
        from .pluto import PlutoDataset as DSet
    else:
        raise ImportError("No suitable Dataset type found for '%s'", root)
    return DSet(root)


class AbstractDataset:
    # This is the abstract dataset class from which any
    # project-specific dataset class should inherit
    def __init__(self):
        self.grids = {}
        self.times = {}
        self.timeSeries = {}
        self.fields = {}
        self.particles = {}
        self.aliases = []
        self.calc_vars = {}
        self.units = {}

class Time:
    # This class handles an array of points in time
    def __init__(self, data=None):
        self.data = data

    def __getitem__(self, n):
        return self.data[n]

    def push_back(self, t):
        self.data = np.append(self.data, t)

    def set(self, data):
        self.data = np.array(data)

    def closest_time(self, t):
        return self.data[self.closest_idx(t)]

    def closest_idx(self, t):
        # Assume linear spacing
        beg = self.data[0]
        end = self.data[-1]
        guess = int((t-beg)/(end-beg)*len(self.data))
        k = 3
        indices = np.arange(guess-k, guess+k+1)
        sli = self.data.take(indices, mode="wrap")
        # Try to find t in the vicinity of guess
        # First make a bool array to try if its inbetween two
        # elements of the slice
        inbetween = (t>=sli[:-1]) & (t<=sli[1:])
        if inbetween.any():
            return guess -k + np.argmin(np.abs(t - sli))
        # If no index was found in the vicinity search globaly
        return np.argmin(np.abs(self.data-t))

class Field:
    # This class holds data for a variable from simulation
    # output or a derived variable.
    def __init__(self, name, grid = None, resource = None, data=None, unitSys=None):
        self.grid = grid
        self.data = data
        self.name = name
        self.resource = resource
        self.unitSys = unitSys

    def load(self):
        raise NotImplementedError("load function of abstract class called. Need to define this function for the codespecific Field class")


class TimeSeries:
    """ Stucture to hold data at different points in time.
    The data can be scalar variables or fields defined on a grid
    for each point in time. """
    def __init__(self, name, resource = None, time = None, data = None):
        self.name = name
        if time is not None and data is not None:
            self.__dict__['data'] = data
            self.__dict__['time'] = time
        self.resource = resource

    def load(self, varname = None, n=None ):
        raise NotImplementedError("load function of abstract class called. Need to define this function for the codespecific dataset")

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get('data', n=key)
        else:
            raise KeyError("Only integer indices are supported but got n = '{}'.".format(key))

    def get(self, varname, n = None ):
        if n is None:
            try:
                return self.__dict__[varname]
            except KeyError:
                self.load(varname)
                return self.__dict__[varname]
        else:
            try:
                return self.__dict__[varname][n]
            except KeyError:
                self.load(varname, n)
                return self.__dict__[varname][n]

    def _time(self, n = None):
        return self.get('time', n=n)

    def _data(self, n = None ):
        return self.get('data', n=n)

    def between(self, tmin = -np.inf, tmax = np.inf):
        """ Extract time and data for the interval [tmin, tmax] """
        mask = np.logical_and( self.__dict__['time'] >= tmin, self.__dict__['time'] <= tmax)
        return (self._time(n = mask), self._data(n = mask))

    def truncate(self, n=None):
        """ Truncate the time series at index n.
        If n is not given, the minimum length of time and data vectors is used. """
        if n is None:
            n = min(len(self.data), len(self.time))
        if max(len(self.data), len(self.time)) > n:
            self.time = self.time[:n]
            self.data = self.data[:n]


class FieldTimeSeries(TimeSeries):
    def __init__(self, name, resource, time, grid, unitSys=None, Field=Field):
        super().__init__(name, resource)
        self.time = time
        self.data = []
        self.unitSys = unitSys
        for f in self.resource:
            self.data.append(Field(name, grid=grid, resource=f, unitSys=unitSys))

    def get(self, varname, n = None ):
        self.load(n)
        return self.__dict__[varname][n]

    def load(self, n):
        self.data[n].load()
