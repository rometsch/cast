#----------------------------------------------------------------------
#	This file provides a generic dataset class
#
#	Author	: Thomas Rometsch (thomas.rometsch@student.uni-tuebingen.de)
#	Date	: 2017-12-13
#----------------------------------------------------------------------

import numpy as np
from collections import OrderedDict as ODict
from .getter import Getter
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

class Particle:
	def __init__(self, name=None, resource=None, data={}):
		self.name = name
		self.resource = resource
		self.data = data

	def load(self, varname = None, n=None ):
		raise NotImplementedError("load function of abstract class called. Need to define this function for the codespecific dataset")

	def __getitem__(self, varname):
		try:
			return self.data[varname]
		except KeyError:
			self.load()
			return self.data[varname]


class Field:
	# This class holds data for a variable from simulation
	# output or a derived variable.
	def __init__(self, name, grid = None, resource = None):
		self.grid = grid
		self.data = None
		self.name = name
		self.resource = resource

	def load(self):
		raise NotImplementedError("load function of abstract class called. Need to define this function for the codespecific Field class")


class TimeSeries:
	""" Stucture to hold data at different points in time.
	The data can be scalar variables or fields defined on a grid
	for each point in time. """
	def __init__(self, name, resource = None):
		self.name = name
		self.resource = resource

	def load(self, varname = None, n=None ):
		raise NotImplementedError("load function of abstract class called. Need to define this function for the codespecific dataset")

	def __getitem__(self, key):
		if isinstance(key, int):
			return self.get('data', n=key)
		else:
			raise KeyError("Only integer indices are supported.")

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

	def time(self, n = None):
		return self.get('time', n=n)

	def data(self, n = None ):
		return self.get('data', n=n)


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
