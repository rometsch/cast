#----------------------------------------------------------------------
#	A dataset wrapper for fargo3d simulations
#
#	Author	: Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#	Date	: 2018-07-10
#----------------------------------------------------------------------

import re
import os
import numpy as np
from . import grid
from . import units
from .units import Dimension as Dim
from .dataset import *
import astropy.units

""" Datafiles produced by fargo3d.
Filenames are given as regex expression to extract from file list."""
datafile_pattern = ".*\.dat"

outputdir_indicators = ["variables.par", "dimensions.dat"]

time_files = { "fine": "time.dat",
			   "coarse" : "timeCoarse.dat"}

scalar_files = ["mass.dat", "momx.dat", "momy.dat", "momz.dat", ]

collections = ["time.dat"]

# Look for planet related files and extract the planet number
# from the filename using (\d+) group in regex pattern
particle_file_pattern = ["bigplanet(\d+)\.dat", "orbit(\d+)\.dat", "a._planet_(\d+)\.dat"]

# capture output data (name, number) for fields with filenames like "gasdens7.dat"
field_pattern = "([a-z]*[A-Z]*)([0-9]+)\.dat"

known_units = {
	'mass' : Dim(M=1),
	'mom.' : Dim(M=1, L=2, T=-1),
	'gasdens' : Dim(M=1, L=-3),
	'gasv.' : Dim(L=1, T=-1),
	'gasenergy' : Dim(M=1, L=2, T=-2)
}

def parse_text_header(fpath):
	with open(fpath, 'r') as f:
		header = []
		for l in f:
			l = l.strip()
			if l[0] != '#':
				break
			header.append(l)
	return parse_text_header_v1(header)

def parse_text_header_v1(header):
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

def centered_coordinates(xInterface):
	x = 0.5*(xInterface[1:] + xInterface[:-1])
	dx = xInterface[1:] - xInterface[:-1]
	return (x, dx)

def parse_Fargo3d_grid(datadir, unitSys=None):
	fpath = os.path.join(datadir, "dimensions.dat")
	data = np.genfromtxt(fpath)
	N = [int(data[k]) for k in [6,7,8]]
	NGH = [int(data[k]) for k in [13,11,12]]
	if unitSys is not None:
		units = [ 1, unitSys['L'], 1]
	else:
		units = [ 1, 1, 1]
	domain_data = {}
	for k, dom, flabel, unit in zip(range(3), ["phi", "r", "theta"], ["x", "y", "z"], units):
		if N[k] > 1:
			domain_full = np.genfromtxt(os.path.join(datadir, "domain_" + flabel + ".dat"))
			if NGH[k] > 0:
				xInterface = domain_full[ NGH[k] : -NGH[k]]
			else:
				xInterface = domain_full
			x, dx = centered_coordinates(xInterface)
			if dom=="phi": # correct the definition of theta in [-pi,pi]
				x += np.pi
			domain_data[dom] = x*unit
			domain_data["d"+dom] = dx*unit
			if len(domain_data[dom]) != N[k]:
				raise ValueError("Domain info files currupted:\
					N{} = {} != {} = Ntot - 2*Nghost".format(dom, len(domain_data[dom]), N[k]))
	return grid.SphericalRegularGrid(**domain_data)

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

class Fargo3dParticle(Particle):
	def __init__(self, name, resource=None, unitSys=None):
		super().__init__(name=name, resource = {})
		self.unitSys = unitSys
		if resource:
			self.add_resource(resource)

	def add_resource(self, res):
		fname = os.path.basename(res)
		if re.match("bigplanet\d+\.dat", fname):
			self.resource['planet'] = res
		elif re.match("orbit\d+\.dat", fname):
			self.resource['orbit'] = res
		else:
			rs = re.match("a(\w+)_planet_\d\.dat", fname)
			if rs:
				self.resource["a{}".format(rs.groups()[0])] = res

	def load(self):
		if "planet" in self.resource:
			data = np.genfromtxt(self.resource["planet"])
			names = ["TimeStep", "x1", "x2", "x3", "v1",
					 "v2", "v3", "mass", "time", "OmegaFrame"]
			for k, name in enumerate(names):
				self.data[name] = data[:,k]

			for v in ["x1", "x2", "x3"]:
				self.data[v] *= self.unitSys['L']

			for v in ["v1", "v2", "v3"]:
				self.data[v] *= self.unitSys['L']/self.unitSys['T']

			self.data["mass"] *= self.unitSys['M']
			self.data["time"] *= self.unitSys['T']
			self.data["OmegaFrame"] *= self.unitSys['T']**(-1)

		if "orbit" in self.resource:
			data = np.genfromtxt(self.resource["orbit"])
			names = ["time", "e", "a", "MeanAnomaly",
					 "TrueAnomaly", "Periastron", "XPosAngle",
					 "i", "AscendingNode", "XYPerihelion"]
			for k, name in enumerate(names):
				self.data[name] = data[:,k]

			self.data["time"] *= self.unitSys['T']
			self.data["a"]	  *= self.unitSys['L']
			for v in ["MeanAnomaly", "TrueAnomaly"]:
				self.data[v]	*= self.unitSys['T']**(-1)

		for v in ["ax", "ay", "az"]:
			if v in self.resource:
				data = np.genfromtxt(self.resource[v])[:,1]
				self.data[v] = data*self.unitSys['L']*self.unitSys['T']**(-2)

class Fargo3dField(Field):
	def __init__(self, *args, unitSys=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.unitSys = unitSys

	def load(self):
		if self.data is None:
			self.data = np.fromfile(self.resource).reshape(self.grid.shape).transpose()
			if self.unitSys is not None and self.name is not None:
				self.data = self.data*self.unitSys.find(self.name)
			if self.unitSys is not None and self.name == 'gasdens' and self.grid.dim == 2:
				self.data = self.data*self.unitSys['L']


class Fargo3dDataset(AbstractDataset):
	def __init__(self, rootdir):
		super().__init__()
		self.rootdir = rootdir
		self.find_datadir()
		self.units = units.parse_code_units_file(self.datadir)
		for key in known_units:
			self.units.register(key, known_units[key])
		self.find_grids()
		self.find_times()
		self.find_datafiles()
		self.find_scalars()
		self.find_collections()
		self.find_particles()
		self.find_fields()

	def find_datadir(self):
		# Look for an output dir inside the rootdir
		self.datadir = find_dir_containing(outputdir_indicators, self.rootdir)

	def find_grids(self):
		self.grids["full"] = parse_Fargo3d_grid(self.datadir, unitSys=self.units)

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
						self.particles[name] = Fargo3dParticle(name, resource=fpath, unitSys=self.units)
					else:
						self.particles[name].add_resource(fpath)

	def find_times(self):
		for name in time_files:
			fpath = os.path.join(self.datadir, time_files[name])
			if time_files[name] in os.listdir(self.datadir):
				if name == "fine":
					data = np.genfromtxt(fpath, usecols=(3))*self.units['T']
				elif name == "coarse":
					data = np.genfromtxt(fpath, usecols=(2))*self.units['T']
				else:
					raise ValueError("Don't know how to parse time info for '{}'".format(time_files[name]))
				self.times[name] = Time(data=data)

	def find_fields(self):
		matches = [m for m in (re.match(field_pattern, f) for f in self.datafiles) if m is not None]
		fields = {}
		for m in matches:
			name, nr = m.groups()
			if name not in fields:
				fields[name] = [m]
			else:
				fields[name].append(m)
		for name in fields:
			fields[name] = sorted(fields[name], key=lambda item: item.groups()[1])
			files = [os.path.join(self.datadir,m.string) for m in fields[name]]
			self.fields[name] = FieldTimeSeries(name, files, self.times['coarse'],
												self.grids['full'], unitSys=self.units,
												Field = Fargo3dField)
