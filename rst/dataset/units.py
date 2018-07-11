#----------------------------------------------------------------------
#	Handling units for datasets.
#   Wrapper to astropy.units
#
#	Author 	: Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#	Date	: 2018-07-11
#----------------------------------------------------------------------

import astropy.units as u
import re

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
	def __init__(self, bases, powers):
		self.bases = bases
		self.powers = powers

	def toUnit(self, us):
		rv = u.Unit(1)
		for b,p in zip(self.bases, self.powers):
			rv *= u.Unit(us[b])**p
		return u.Unit(rv)

class UnitSystem:
	def __init__(self, units = {}):
		self.knownUnits = {}
		self.units = {}
		for key in units:
			self.units[key] = self.parse(units[key])

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
