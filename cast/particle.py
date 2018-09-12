#----------------------------------------------------------------------
#   Classes to handle nbody particles/bodies/planets
#
#   Author  : Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#   Date    : 2018-09-03
#----------------------------------------------------------------------

import numpy as np
import os
from .dataset import TimeSeries
import astropy.units as u
import astropy.constants as const

class Particle:
    def __init__(self, name=None, resource=None, data=None, unitSys=None):
        self.name = name
        self.resource = resource
        self.unitSys = unitSys
        self.data = data if data is not None else {}

    def load(self, varname = None, n=None ):
        raise NotImplementedError("load function of abstract class called. Need to define this function for the codespecific dataset")

    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError:
            self.load()
            try:
                return self.data[key]
            except KeyError:
                try:
                    getattr(self, "_{}".format(key))()
                    return self.data[key]
                except AttributeError:
                    raise AttributeError("Can't load or calculate '{}'.".format(key))
            

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

    def calc_orbital_elements(self, Mstar=1*u.solMass):
        """ Calculate orbital elements using postition (x), velocities (v) and stellar mass (Mstar) """
        mu = Mstar*const.G

        x1 = self['x1'].data
        x2 = self['x2'].data
        x3 = self['x3'].data
        v1 = self['v1'].data
        v2 = self['v2'].data
        v3 = self['v3'].data

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
        
        self.data['E'] = TimeSeries('specific energy', time=self['time'], data=E)
        self.data['a'] = TimeSeries('semi major axis', time=self['time'], data=a)
        self.data['e'] = TimeSeries('eccentricity', time=self['time'], data=e)
        self.data['i'] = TimeSeries('inclination', time=self['time'], data=i)
        self.data['AscendingNode'] = TimeSeries('longitude of ascending node', time=self['time'], data=Omega)
        self.data['Periastron'] = TimeSeries('argument of periastron', time=self['time'], data=omega)
        self.data['TrueAnomaly'] = TimeSeries('true anomaly', time=self['time'], data=nu)
        self.data['xi'] = TimeSeries('angle to line of nodes', time=self['time'], data=xi)

    def calc_accelerations_RTN(self):
        """ Calculate the acceleration in radial, tangengial and normal directions """
        Torb = self['Torb'][0]
        t = self['time'].to(Torb)

        a1 = self['a1'].data
        a2 = self['a2'].data
        a3 = self['a3'].data
        x1 = self['x1'].data
        x2 = self['x2'].data
        x3 = self['x3'].data
        v1 = self['v1'].data
        v2 = self['v2'].data
        v3 = self['v3'].data

        r = np.sqrt(x1**2 + x2**2 + x3 **2)
        v = np.sqrt(v1**2 + v2**2 + v3 **2)

        r1 = x1/r
        r2 = x2/r
        r3 = x3/r

        t1 = v1/v
        t2 = v2/v
        t3 = v3/v

        h1 = x2*v3 - x3*v2
        h2 = x3*v1 - x1*v3
        h3 = x1*v2 - x2*v1
        h = np.sqrt(h1**2 + h2**2 + h3**2)
        n1 = h1/h
        n2 = h2/h
        n3 = h3/h

        self.data['ar'] = a1*r1 + a2*r2 + a3*r3
        self.data['at'] = a1*t1 + a2*t2 + a3*t3
        self.data['an'] = a1*n1 + a2*n2 + a3*n3

    def _Torb(self):
        """ Calculate orbital period """
        Torb = np.sqrt(self['a'].data**3*4*np.pi**2/(const.G*(1*u.solMass+self['mass'][0]) )).to(u.yr)
        self.data['Torb'] = TimeSeries('orbital period', time=self.data['time'], data=Torb)
        
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
