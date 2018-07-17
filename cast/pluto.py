#----------------------------------------------------------------------
#	A dataset wrapper for pluto simulations
#
#	Author 	: Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#	Date	: 2018-07-12
#----------------------------------------------------------------------

from cast.dataset import *
from . import units
from .units import Dimension as Dim
import re
import os
import numpy as np
import astropy.units


""" Datafiles produced by pluto.
Filenames are given as regex expression to extract from file list."""
datafile_pattern = ".*\.dat"

outputdir_indicators = ["grid.out", "dbl.out"]

scalar_files = []

collections = ["analysis_values.dat"]

# Look for planet related files and extract the planet number
# from the filename using (\d+) group in regex pattern
particle_file_pattern = []

known_units = {
	"niter"    : Dim(),
	"M_DISK"   : Dim(M=1),
	"KE_R"     : Dim(M=1, L=2, T=-2),
	"KE_TH"    : Dim(M=1, L=2, T=-2),
	"KE_PHI"   : Dim(M=1, L=2, T=-2),
	"RHO_MIN"  : Dim(M=1, L=-3),
	"RHO_MAX"  : Dim(M=1, L=-3),
	"J_DISK_0" : Dim(M=1, L=2, T=-1),
	"J_DISK_1" : Dim(M=1, L=2, T=-1),
	"J_DISK_2" : Dim(M=1, L=2, T=-1),
	"F_0"      : Dim(M=1, L=1, T=-2),
	"F_1"      : Dim(M=1, L=1, T=-2),
	"F_2"      : Dim(M=1, L=1, T=-2)
}

def find_pluto_log(rootdir):
	try:
		rv = os.path.join(find_dir_containing("pluto.log", rootdir), "pluto.log")
	except FileNotFoundError:
		#rv = None
		raise
	print(rv)
	return rv

def extract_pluto_log_units(datadir):
	fpath = find_pluto_log(datadir)
	if fpath is None:
		rv = units.UnitSystem()
	else:
		foundUnits = {}
		with open(fpath, 'r') as f:
			for l in f:
				# match to lines like: [Density]:      4.249e-09 (gr/cm^3), 2.540e+15 (1/cm^3)
				rs = re.match("\s*\[(\w+)\]\:\s*([0-9\.e\-\+]+).*\((.+)\)",l.split(',')[0].strip())
				if rs:
					rs = rs.groups()
					name = rs[0]
					value = rs[1]
					unitstring = rs[2].replace('^', '').replace('gr','g').replace('sec', 's')
					foundUnits[rs[0]] = float(value)*astropy.units.Unit(unitstring)
		baseunits = {
			"L" : foundUnits['Length'],
			"T" : foundUnits['Time'],
			"Th" : foundUnits['Temperature'],
			"M" : foundUnits['Density']*foundUnits['Length']**3
			}
		rv = units.UnitSystem(baseunits)
	return rv

def parse_text_header(fpath):
	with open(fpath, 'r') as f:
		header = []
		for l in f:
			l = l.strip()
			if l[0] != '#':
				break
			header.append(l)
	return parse_text_header_pluto(header)

def parse_text_header_pluto(header):
	names = []
	for l in header:
		l = l.strip('#').strip()
		if l[:4] == "vars":
			l = l[4:].lstrip().lstrip(":")
			names = [s.strip() for s in l.split('\t')]
	timecol = None
	for n, name in enumerate(names):
		if name in ['time', 'simulation time']:
			timecol = n
	return names, timecol

class ScalarTimeSeries(TimeSeries):
    def __init__(self, time=None, data=None, datafile=None, name = None, unitSys = None):
        self.time = time
        self.data = data
        self.datafile = datafile
        self.name = name
        self.unitSys = unitSys

    def load(self, *args, **kwargs):
        pass

class PlutoParticle(Particle):
	def __init__(self, name, resource=None, unitSys=None):
		pass

class PlutoDataset(AbstractDataset):
	def __init__(self, rootdir):
		super().__init__()
		self.rootdir = rootdir
		self.find_datadir()
		self.units = extract_pluto_log_units(self.rootdir)
		for key in known_units:
			self.units.register(key, known_units[key])
		self.find_datafiles()
		self.find_scalars()
		self.find_collections()
		self.find_particles()

	def find_datadir(self):
		# Look for an output dir inside the rootdir
		self.datadir = find_dir_containing(outputdir_indicators, self.rootdir)

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
		names, timecol = parse_text_header(fpath)
		if timecol is None:
			raise TypeError("Could not find time info for constructing time series in '{}' with names '{}'.".format(fpath, names))
		data = np.genfromtxt(fpath)
		time = data[:,timecol]*self.units['time']
		for n, name in enumerate(names):
			if n == timecol:
				continue
			vardata = data[:,n]*self.units.find(name)
			self.timeSeries[name] = ScalarTimeSeries(time = time, data = vardata, name = name)

	def find_particles(self):
		for f in self.datafiles:
			for p in particle_file_pattern:
				rs = re.match(p, f)
				if rs:
					fpath = os.path.join(self.datadir, f)
					name = rs.groups()[0]
					if name not in self.particles:
						self.particles[name] = PlutoParticle(name, resource=fpath, unitSys=self.units)
					else:
						self.particles[name].add_resource(fpath)
