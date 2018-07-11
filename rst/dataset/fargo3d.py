#----------------------------------------------------------------------
#	A dataset wrapper for fargo3d simulations
#
#	Author 	: Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#	Date	: 2018-07-10
#----------------------------------------------------------------------

from dataset import *
import units
import re
import os
import numpy as np


""" Datafiles produced by fargo3d.
Filenames are given as regex expression to extract from file list."""
datafile_pattern = ".*\.dat"

outputdir_indicators = ["variables.par", "dimensions.dat"]

time_files = ["time.dat", "timeCoarse.dat"]

scalar_files = ["mass.dat", "momx.dat", "momy.dat", "momz.dat", ]

collections = ["time.dat"]

known_units = {
	'mass' : units.Dimension(['mass'], [1]),
	'mom.' : units.Dimension(['mass', 'length', 'time'], [1, 2, -1])
}

def fargo_code_units(datadir):
    cu = {}
    try:
        with open( os.path.join(datadir, 'units.dat')) as unitfile:
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
        # Fall back to a wild guess
        cu['length'] = 'meter'
        cu['mass'] = 'kg'
        cu['time'] = 'year'
    return units.UnitSystem(cu)

class ScalarTimeSeries(TimeSeries):
    def __init__(self, time=None, data=None, datafile=None, name = None, unitSys = None):
        self.time = time
        self.data = data
        self.datafile = datafile
        self.name = name
        self.unitSys = unitSys

    def load(self, *args, **kwargs):
        data = np.genfromtxt(self.datafile)
        self.time = data[:,0]

        for n in range(1, data.shape[1]):
            self.data = data[:,1]

        if self.unitSys is not None:
            self.time *= self.unitSys['time']
            if self.name is not None:
                self.data *= self.unitSys.find(self.name)


class Fargo3dDataset(Dataset):

	def __init__(self, rootdir):
		super().__init__()
		self.rootdir = rootdir
		self.find_datadir()
		self.units = fargo_code_units(self.datadir)
		for key in known_units:
			self.units.register(key, known_units[key])
		self.find_datafiles()
		self.find_scalars()
		self.find_collections()

	def find_datadir(self):
		# Look for an output dir inside the rootdir
		for dirname, dirnames, filenames in os.walk(self.rootdir):
			for subdirname in dirnames:
				path = os.path.join(dirname, subdirname)
				if all(s in os.listdir(path) for s in outputdir_indicators):
					self.datadir = path

	def find_datafiles(self):
		""" Search the datadir for datafiles."""
		matches = (re.match(datafile_pattern, s) for s in os.listdir(self.datadir))
		self.datafiles = [m.string for m in matches if m is not None]

	def find_scalars(self):
		""" Find all time info."""
		for s in scalar_files:
			if s in self.datafiles:
				name = s[:-4]
				self.timeSeries[name] = ScalarTimeSeries(os.path.join(self.datadir, s)
														 , name=name, unitSys=self.units)

	def find_collections(self):
		""" Find files containing multiple time series and parse them. """
		for c in collections:
			if c in self.datafiles:
				self.add_collection(c)

	def add_collection(self, collection):
		fpath = os.path.join(self.datadir, collection)
		names, timecol = parse_header(fpath)
		if timecol is None:
			raise TypeError("Could not find time info for constructing time series in '{}' with names '{}'.".format(fpath, names))
		data = np.genfromtxt(fpath)
		time = data[:,timecol]*self.units['time']
		for n, name in enumerate(names):
			if n == timecol:
				continue
			vardata = data[:,n]*self.units.find(name)
			self.timeSeries[name] = ScalarTimeSeries(time = time, data = vardata, name = name)

def parse_header(fpath):
	with open(fpath, 'r') as f:
		header = []
		for l in f:
			l = l.strip()
			if l[0] != '#':
				break
			header.append(l)
	return parse_v1_header(header)

def parse_v1_header(header):
	names = []
	for l in header:
		l = l.strip('#').strip()
		if l[:6] == "Syntax":
			l = l[6:].lstrip().lstrip(":")
			names = [s.strip() for s in l.split('<tab>')]
	timecol = None
	for n, name in enumerate(names):
		if name in ['time', 'simulation time']:
			timecol = n
	return names, timecol