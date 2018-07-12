#----------------------------------------------------------------------
#	Handling units for datasets.
#   Wrapper to astropy.units
#
#	Author 	: Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#	Date	: 2018-07-11
#----------------------------------------------------------------------

import astropy.units as u
import re
import os

defaults_astropy = {
	'L' : u.meter,		# length
	'T' : u.second,		# time
	'M' : u.kg,			# mass
    'C' : u.ampere,		# current
    'Th' : u.Kelvin,	# temperature (Theta)
    'Ad' : u.radian,    # angular distance
    'SA' : u.steradian, # solid angle
    'I' : u.candela,	# luminous intensity
    'Mag' : u.mag,		# stellar magnitude
    'A' : u.mole,		# amount of substance
    'N' : u.photon		# photon count
}

defaults = {
	'L'   : u.Unit(1),	# length
	'T'   : u.Unit(1),	# time
	'M'   : u.Unit(1),	# mass
    'C'   : u.Unit(1),	# current
    'Th'  : u.Unit(1),	# temperature (Theta)
    'Ad'  : u.Unit(1),	# angular distance
    'SA'  : u.Unit(1),	# solid angle
    'I'   : u.Unit(1),	# luminous intensity
    'Mag' : u.Unit(1),	# stellar magnitude
    'A'   : u.Unit(1),	# amount of substance
    'N'   : u.Unit(1)	# photon count
}

alias = {
	'length'		: 'L'  ,	# length
	'time'			: 'T'  ,	# time
	'mass'			: 'M'  ,	# mass
    'current'		: 'C'  ,	# current
    'temperature'	: 'Th' ,	# temperature (Theta)
    'intensity'		: 'Lum',	# luminous intensity
    'amount'		: 'A'  ,	# amount of substance
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
		rv = 1 #u.Unit(1)
		for b,p in zip(self.bases, self.powers):
			rv *= u.Unit(us[b])**p
		return u.Unit(rv)

def parse_code_units_file(datadir, defaults = defaults):
    cu = {}
    try:
        with open( os.path.join(datadir, 'units.inf')) as unitfile:
            for l in unitfile:
                l = l.strip()
                if l[0] == "#":
                    continue
                l = l.split('\t')
                ignore_chars = ['', '?']
                if l[1] in ignore_chars or l[2] in ignore_chars:
                    continue
                cu[l[0]] = l[1:]
    except OSError:
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
		rv = u.Unit(1)
		for key in self.knownUnits:
			if re.match(key, name):
				rv = self.knownUnits[key]
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

		if not isinstance(rv, u.core.UnitBase):
			raise TypeError("'{}' could not be cast to a unit.".format(un))
		return rv
