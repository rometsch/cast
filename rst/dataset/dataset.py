#----------------------------------------------------------------------
#	This file provides a generic dataset class
#
#	Author 	: Thomas Rometsch (thomas.rometsch@student.uni-tuebingen.de)
#	Date	: 2017-12-13
#----------------------------------------------------------------------

import unicodedata
import numpy as np
from collections import OrderedDict as ODict
from getter import Getter

class Dataset:
	# This is the abstract dataset class from which any
	# project-specific dataset class should inherit
	def __init__(self):
		self.grids = []
		self.times = []
		self.timeSeries = {}
		self.particles = {}
		self.aliases = []
		self.calc_vars = {}
		self.units = {}

class Time:
	# This class handles an array of points in time
	def __init__(self, data=None):
		if data is not None:
			self.data = np.array(data)
		else:
			self.data = np.array([])

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
	def __init__(self, data, grid=None):
		self.data = data
		self.grid = grid

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
			return (self.get('time', n=key), self.get('data', n=key))
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
