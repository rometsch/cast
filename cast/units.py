#----------------------------------------------------------------------
#   Handling units for datasets.
#   Wrapper to astropy.units
#
#   Author  : Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#   Date    : 2018-07-11
#----------------------------------------------------------------------

import astropy.units as u
import re
import os

defaults_astropy = {
    'L' : u.meter,      # length
    'T' : u.second,     # time
    'M' : u.kg,         # mass
    'C' : u.ampere,     # current
    'Th' : u.Kelvin,    # temperature (Theta)
    'Ad' : u.radian,    # angular distance
    'SA' : u.steradian, # solid angle
    'I' : u.candela,    # luminous intensity
    'Mag' : u.mag,      # stellar magnitude
    'A' : u.mole,       # amount of substance
    'N' : u.photon      # photon count
}

defaults = {
    'L'   : 1,  # length
    'T'   : 1,  # time
    'M'   : 1,  # mass
    'C'   : 1,  # current
    'Th'  : 1,  # temperature (Theta)
    'Ad'  : 1,  # angular distance
    'SA'  : 1,  # solid angle
    'I'   : 1,  # luminous intensity
    'Mag' : 1,  # stellar magnitude
    'A'   : 1,  # amount of substance
    'N'   : 1   # photon count
}

unitsymbols = ['L','T','M','C','Th','Ad','SA','I','Mag','A','N']

alias = {
    'length'        : 'L'  ,    # length
    'time'          : 'T'  ,    # time
    'mass'          : 'M'  ,    # mass
    'current'       : 'C'  ,    # current
    'temperature'   : 'Th' ,    # temperature (Theta)
    'intensity'     : 'Lum',    # luminous intensity
    'amount'        : 'A'  ,    # amount of substance
   }


class Dimension:
    def __init__(self, bases=None, powers=None, **kwargs):
        if bases and powers:
            self.bases = bases
            self.powers = powers
        else:
            self.bases = []
            self.powers = []
            for s in unitsymbols:
                if s in kwargs:
                    self.bases.append(s)
                    self.powers.append(kwargs[s])



    def toUnit(self, us):
        rv = 1
        for b,p in zip(self.bases, self.powers):
            rv *= u.Unit(us[b])**p
        if rv != 1:
            rv = u.Unit(rv)
        return rv

    def __repr__(self):
        rv = "{}: '".format(self.__class__.__name__) + " ".join(("{}{}".format(b,p) for b,p in zip(self.bases, self.powers))) + "'"
        return rv

def parse_code_units_file(datadir, defaults = defaults):
    cu = {}
    fpath = None
    for fname in ['units.inf', 'units.dat']:
        if fname in os.listdir(datadir):
            fpath = os.path.join(datadir, fname)
            break
    try:
        with open(fpath) as unitfile:
            for l in unitfile:
                l = l.strip()
                if l[0] == "#":
                    continue
                l = l.split('\t')
                ignore_chars = ['', '?']
                if l[1] in ignore_chars or l[2] in ignore_chars:
                    continue
                cu[l[0]] = l[1:]
    except (AttributeError, OSError, TypeError):
        print("Warning: Could not find output baseunits in '{}'".format(datadir))
        pass
    return UnitSystem(cu, defaults=defaults)

class UnitSystem:
    def __init__(self, units = {}, defaults=defaults):
        self.knownUnits = {}
        self.defaults = defaults
        self.units = {}
        for key in units:
            if key in alias:
                name = alias[key]
            else:
                name = key
            self.units[name] = self.parse(units[key])

        for key in defaults:
            if key not in self.units:
                self.units[key] = defaults[key]

    def __getitem__(self, key):
        try:
            return self.units[key]
        except KeyError:
            try:
                return self.units[alias[key]]
            except KeyError:
                raise KeyError("Unit '{}' is not known to unit system.".format(key))

    def register(self, regex, unit):
        self.knownUnits[regex] = self.parse(unit)

    def find(self, name):
        rv = 1
        for key in self.knownUnits:
            if re.match(key, name):
                rv = self.knownUnits[key]
        if not isinstance(rv, u.core.UnitBase): #rv is 1:
            print("Warning: No unit found for '{}'".format(name))
        return rv

    def parse(self, un):
        rv = None
        if isinstance(un, u.Unit) or isinstance(un, u.Quantity):
            rv = u.Unit(un)
        elif isinstance(un, str):
            rv = u.Units(un)
        elif (isinstance(un, list) or isinstance(un, tuple)) and len(un) == 2:
            rv = u.Unit(un[0]*u.Unit(un[1]))
        elif isinstance(un, Dimension):
            rv = un.toUnit(self)
        elif un == 1:
            rv = 1

        if not (isinstance(rv, u.core.UnitBase) or rv == 1):
            raise TypeError("'{}' could not be cast to a unit: result was '{}'.".format(un, rv))
        return rv
