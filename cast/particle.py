#----------------------------------------------------------------------
#   Classes to handle nbody particles/bodies/planets
#
#   Author  : Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#   Date    : 2018-09-03
#----------------------------------------------------------------------

import numpy as np
import os
from .dataset import TimeSeries

class Particle:
    def __init__(self, name=None, resource=None, data=None, unitSys=None):
        self.name = name
        self.resource = resource
        self.unitSys = unitSys
        self.data = data if data is not None else {}

    def load(self, varname = None, n=None ):
        raise NotImplementedError("load function of abstract class called. Need to define this function for the codespecific dataset")

    def __getitem__(self, varname):
        try:
            return self.data[varname]
        except KeyError:
            self.load()
            return self.data[varname]

    def truncate(self, t=None):
        """ Truncate all time series to time t (default: the shortest available time series) """
        # Make sure data is loaded
        if not 'time' in self.data:
            self.load()
        # check whether final time is supplied
        if t is None:
            t = self['time'][-1]

        # make sure data and time vectors have the same length
        for key, ts in self.data.items():
            if key == 'time':
                continue
            self.data[key].truncate()

        for key, ts in self.data.items():
            if key == 'time':
                continue
            if self['time'][-1] > ts.time[-1]:
                self.data['time'] = ts.time

        for key, ts in self.data.items():
            if key == 'time':
                continue
            if ts.time[-1] > self['time'][-1]:
                ts.time = self['time']
                ts.data = ts.data[:len(ts.time)]

class Planet(Particle):

    def recalc_orbital_elements(self):
        x1 = p['x1'].data
        x2 = p['x2'].data
        x3 = p['x3'].data
        v1 = p['v1'].data
        v2 = p['v2'].data
        v3 = p['v3'].data

        h1 = x2*v3 - v2*x3
        h2 = x3*v1 - v3*x1
        h3 = x1*v2 - v1*x2

        def norm(a,b,c):
            return np.sqrt(a**2 + b**2 + c**2)

        r = norm(x1, x2, x3)
        v = norm(v1, v2, v3)
        h = norm(h1, h2, h3)

        E = v**2/2 - mu/r
        a = -mu/(2*E)
        e = np.sqrt(1 - h**2/(a*mu))
        i = np.arccos(h3/h).to('degree')
        Omega = np.arctan2(h1, -h2)
        xi = np.arctan2( x3/np.sin(i), x1*np.cos(Omega) + x2*np.sin(Omega))
        nu = np.arccos( (a*(1-e**2) - r)/(e*r) )
        omega = xi - nu
        
        self.data['E'] = TimeSeries('specific energy', time=self.time, data=E)
        self.data['a'] = TimeSeries('semi major axis', time=self.time, data=a)
        self.data['e'] = TimeSeries('eccentricity', time=self.time, data=e)
        self.data['i'] = TimeSeries('inclination', time=self.time, data=i)
        self.data['AscendingNode'] = TimeSeries('longitude of ascending node', time=self.time, data=Omega)
        self.data['Periastron'] = TimeSeries('argument of periastron', time=self.time, data='omega')
        self.data['TrueAnomaly'] = TimeSeries('true anomaly', time=self.time, data=nu)
        self.data['xi'] = TimeSeries('angle to line of nodes', time=self.time, data=xi)

class PlanetSystem:
    def __init__(self, planets):
        self.planets = planets
        self._planets_by_name = {p.name : p for p in self.planets}

    def __getitem__(self, key):
        try:
            return self.planets[key]
        except TypeError:
            try:
                return self._planets_by_name[key]
            except KeyError:
                raise KeyError("{} not in planets nor is it a planet's name".format(key))
