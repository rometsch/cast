#----------------------------------------------------------------------
#	This file provides a generic getter class that supports
#	aliases for variables and can calculate new data from existing
#	data if suitable functions are provided.
#
#	Author 	: Thomas Rometsch (thomas.rometsch@student.uni-tuebingen.de)
#	Date	: 2017-12-13
#----------------------------------------------------------------------

class Getter:
	# A fancy getter that provides data from a data dict
	# and calculates data if a suitable function is
	# provided.
	def __new__(cls):
		# Give the getter the neccecary attributes
		ins = super().__new__(cls)
		ins.data = {}
		ins.__aliases = {}
		ins.__provide_functions = {}
		ins.__max_get_depth = 2
		return ins

#	def get(self, key, **kwargs):
#		if key in self.data:
#			return self.data[key]

	def __getitem__(self, key):
	# Return values from data if existend
	# otherwise fall back to fancy getter
		if key in self.data:
			return self.data[key]
		else:
			return self.__fancy_get(key)

	def __fancy_get(self, key, depth=0):
		# This getter first tries to provide
		# the quantity via an alias for which data
		# might be present in self.data.
		# Then it attempts to calculate/provide it
		# if a suitable function is registered
		# and finally it recursively tries the aliases
		# (this also tries to calculate the alias)
		# up to a depth defined by self.__max_get_depth.
		if depth > self.__max_get_depth:
			raise RecursionError("Reached max depth of {}".format(self.__max_get_depth))

		# First try the aliases
		if key in self.__aliases:
			for alias in self.__aliases[key]:
				if alias in self.data:
					return self.data[alias]

		# Then try to calculate it
		if key in self.__provide_functions:
			val = self.__provide_functions[key]()
			self[key] = val
			return val

		# Finally launch getters for all aliases
		if key in self.__aliases:
			for alias in self.__aliases[key]:
				try:
					return self.__fancy_get(key, depth=depth+1)
				except RecursionError:
					pass

		# If all fails
		raise KeyError("No data for **{}** : fancy getter failed".format(key))

	def __setitem__(self, key, val):
		self.data[key] = val

	def register_getter_alias(self, name, alter, mutual=False):
		# Registers **alter** as an alias for **name**
		# If mutual is set, the alias is also set reversed.
		if alter in self.__aliases:
			self.__aliases[alter].append(name)
		else:
			self.__aliases[alter] = [name]
		if mutual:
			self.register_getter_alias(alter, name)

	def register_provide_function(self, var, func):
		# Register a function with which to provide
		# data for the **var**.
		# The function needs to return the data.
		self.__provide_functions[var] = func
